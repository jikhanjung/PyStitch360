"""영상 구간 원통 파노라마 스티칭 프로토타입.

첫 프레임 쌍에서 정합(stitch_still.setup_alignment)을 1회 추정하고,
remap 테이블·게인·페더 가중치를 캐싱한 뒤 프레임 루프는
cv2.remap 2회 + 가중합만 수행. 출력은 ffmpeg 파이프로 인코딩.
오디오는 좌측 카메라 트랙을 사용.

사용법:
  python stitch_video.py L.mp4 R.mp4 --offset 0.068 --start 300 --duration 10 -o out.mp4
"""
import argparse
import subprocess
import time

import cv2
import numpy as np

from stitch_still import (
    load_lens_profile,
    setup_alignment,
    build_cylindrical_maps,
    compute_gains,
    seam_weights,
)


def open_at(path, t_sec):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("left")
    ap.add_argument("right")
    ap.add_argument("--profile", default="presets/lens_profiles/GoPro_HERO5_Black_Wide_4K_16x9.json")
    ap.add_argument("-o", "--out", default="pano.mp4")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="동기화 오프셋(초): 같은 사건의 R 타임스탬프 - L 타임스탬프")
    ap.add_argument("--start", type=float, default=0.0, help="L 기준 시작 시각(초)")
    ap.add_argument("--duration", type=float, default=10.0, help="처리 길이(초)")
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--roll", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--crf", type=int, default=19)
    args = ap.parse_args()

    K, D, dim = load_lens_profile(args.profile)

    cap_l = open_at(args.left, args.start)
    cap_r = open_at(args.right, args.start + args.offset)
    fps = cap_l.get(cv2.CAP_PROP_FPS)
    ok_l, img_l = cap_l.read()
    ok_r, img_r = cap_r.read()
    assert ok_l and ok_r, "프레임 읽기 실패"
    assert img_l.shape[1] == dim[0], "해상도가 렌즈 프로파일과 다름"

    print("정합 추정 중 (1회)...")
    t0 = time.perf_counter()
    g = setup_alignment(img_l, img_r, K, D, args.pitch, args.roll, args.yaw)
    print(f"  정합 {time.perf_counter()-t0:.1f}s")

    f = g["f"]
    out_w = int((g["yaw1"] - g["yaw0"]) * f) & ~1  # yuv420 짝수 정렬
    out_h = int((np.tan(g["el1"]) - np.tan(g["el0"])) * f) & ~1
    print(f"출력: {out_w}x{out_h} @ {fps:.2f}fps, {args.duration}s")

    print("remap 테이블/가중치 캐싱 중...")
    t0 = time.perf_counter()
    maps, masks = [], []
    for R_cam in (g["R_wl"], g["R_wr"]):
        mx, my = build_cylindrical_maps(K, D, R_cam, out_w, out_h,
                                        g["yaw0"], g["yaw1"], g["el0"], g["el1"])
        # 정수+보간계수 표현으로 변환 (remap 이 더 빠름)
        mx16, my16 = cv2.convertMaps(mx, my, cv2.CV_16SC2)
        maps.append((mx16, my16))
        masks.append(cv2.remap(np.ones((dim[1], dim[0]), np.uint8) * 255, mx, my,
                               cv2.INTER_NEAREST, borderValue=0))
    warp_l = cv2.remap(img_l, *maps[0], interpolation=cv2.INTER_LINEAR)
    warp_r = cv2.remap(img_r, *maps[1], interpolation=cv2.INTER_LINEAR)
    gain_l, gain_r = compute_gains(warp_l, warp_r, masks[0], masks[1])
    gain_l_sc = tuple(gain_l) + (1.0,)   # cv2.multiply 용 스칼라
    gain_r_sc = tuple(gain_r) + (1.0,)
    # 하프라인 수직 심: 심 좌측은 L 카메라, 우측은 R 카메라만 사용
    w_l = seam_weights(masks[0], masks[1], g["yaw0"], g["yaw1"],
                       (g["yaw0"] + g["yaw1"]) / 2)
    w_r = (1.0 - w_l).astype(np.float32)
    print(f"  캐싱 {time.perf_counter()-t0:.1f}s, 게인 L={gain_l.round(3)} R={gain_r.round(3)}")

    n_frames = int(args.duration * fps)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}",
         "-r", f"{fps}", "-i", "-",
         "-ss", f"{args.start}", "-t", f"{args.duration}", "-i", args.left,
         "-map", "0:v", "-map", "1:a?",
         "-c:v", "libx264", "-preset", "fast", "-crf", str(args.crf),
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", args.out],
        stdin=subprocess.PIPE,
    )

    t_read = t_warp = t_blend = t_write = 0.0
    t_start = time.perf_counter()
    done = 0
    for i in range(n_frames):
        t0 = time.perf_counter()
        ok_l, img_l = cap_l.read()
        ok_r, img_r = cap_r.read()
        if not (ok_l and ok_r):
            print(f"  입력 종료 (프레임 {i})")
            break
        t1 = time.perf_counter()
        warp_l = cv2.remap(img_l, *maps[0], interpolation=cv2.INTER_LINEAR)
        warp_r = cv2.remap(img_r, *maps[1], interpolation=cv2.INTER_LINEAR)
        t2 = time.perf_counter()
        warp_l = cv2.multiply(warp_l, gain_l_sc)          # 채널별 게인 (saturate)
        warp_r = cv2.multiply(warp_r, gain_r_sc)
        frame = cv2.blendLinear(warp_l, warp_r, w_l, w_r)  # 네이티브 페더 블렌딩
        t3 = time.perf_counter()
        enc.stdin.write(frame.tobytes())
        t4 = time.perf_counter()
        t_read += t1 - t0
        t_warp += t2 - t1
        t_blend += t3 - t2
        t_write += t4 - t3
        done += 1
        if done % 30 == 0:
            el = time.perf_counter() - t_start
            print(f"  {done}/{n_frames} 프레임, {done/el:.2f} fps")

    enc.stdin.close()
    enc.wait()
    cap_l.release()
    cap_r.release()

    total = time.perf_counter() - t_start
    print(f"\n완료: {done} 프레임 / {total:.1f}s = {done/total:.2f} fps (실시간 대비 x{done/total/fps:.2f})")
    if done:
        print(f"  프레임당: 읽기 {t_read/done*1000:.0f}ms, 워핑 {t_warp/done*1000:.0f}ms, "
              f"블렌딩 {t_blend/done*1000:.0f}ms, 인코더 대기 {t_write/done*1000:.0f}ms")
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
