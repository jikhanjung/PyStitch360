"""파노라마 렌더링: remap 캐싱, 게인 보정, 심/페더 블렌딩."""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import build_cylindrical_maps
from .lens import LensProfile


def compute_gains(warped_l, warped_r, mask_l, mask_r):
    """겹침 영역 채널 평균을 기하평균으로 정렬하는 채널별 게인."""
    overlap = (mask_l > 0) & (mask_r > 0)
    if overlap.sum() < 1000:
        return np.ones(3), np.ones(3)
    mean_l = warped_l[overlap].reshape(-1, 3).mean(axis=0)
    mean_r = warped_r[overlap].reshape(-1, 3).mean(axis=0)
    target = np.sqrt(mean_l * mean_r)
    return target / mean_l, target / mean_r


def feather_weights(mask_l, mask_r, feather_px=120):
    """마스크 경계 거리 기반 페더 가중치 (자동 보정용 저해상도 렌더에 사용)."""
    dist_l = np.minimum(cv2.distanceTransform((mask_l > 0).astype(np.uint8), cv2.DIST_L2, 3), feather_px)
    dist_r = np.minimum(cv2.distanceTransform((mask_r > 0).astype(np.uint8), cv2.DIST_L2, 3), feather_px)
    wsum = dist_l + dist_r
    wsum[wsum == 0] = 1
    return (dist_l / wsum).astype(np.float32)


def seam_weights(mask_l, mask_r, yaw0, yaw1, seam_yaw, feather_px=40):
    """하프라인 수직 심 가중치: 심 좌측은 L, 우측은 R 카메라만 사용.

    심 주변 feather_px 폭만 선형 블렌딩. 한쪽 마스크가 없는 곳은 다른 쪽이 채움.
    """
    h, w = mask_l.shape
    yaws = np.linspace(yaw0, yaw1, w)
    half = feather_px / 2 * (yaw1 - yaw0) / w  # 픽셀 → 라디안
    ramp = np.clip((seam_yaw + half - yaws) / (2 * half), 0.0, 1.0).astype(np.float32)
    w_l = np.tile(ramp, (h, 1)) * (mask_l > 0)
    w_r = np.tile(1.0 - ramp, (h, 1)) * (mask_r > 0)
    wsum = w_l + w_r
    wsum[wsum == 0] = 1
    return (w_l / wsum).astype(np.float32)


def render_pano(imgs, Rs, lens: LensProfile, out_w, out_h, yaw0, yaw1, el0, el1,
                feather_px=120, seam_yaw=None):
    """1회용 파노라마 렌더 (자동 보정 저해상도 패스, 정지 프레임용)."""
    warped, masks = [], []
    for img, R_cam in zip(imgs, Rs):
        mx, my = build_cylindrical_maps(lens, R_cam, out_w, out_h, yaw0, yaw1, el0, el1)
        warped.append(cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=0))
        masks.append(cv2.remap(np.ones(img.shape[:2], np.uint8) * 255, mx, my,
                               cv2.INTER_NEAREST, borderValue=0))
    g_l, g_r = compute_gains(warped[0], warped[1], masks[0], masks[1])
    if seam_yaw is None:
        w_l = feather_weights(masks[0], masks[1], feather_px)[..., None]
    else:
        w_l = seam_weights(masks[0], masks[1], yaw0, yaw1, seam_yaw)[..., None]
    pano = (warped[0].astype(np.float32) * (w_l * g_l)
            + warped[1].astype(np.float32) * ((1 - w_l) * g_r))
    return np.clip(pano, 0, 255).astype(np.uint8)


class Renderer:
    """remap 테이블·가중치를 캐싱한 프레임 렌더러 (미리보기·내보내기 공용).

    scale 로 출력 해상도를 줄일 수 있다 (미리보기용).
    """

    def __init__(self, lens: LensProfile, R_wl, R_wr, yaw0, yaw1, el0, el1,
                 scale=1.0, feather_px=40):
        self.lens = lens
        f = lens.focal * scale
        self.scale = scale
        self.out_w = int((yaw1 - yaw0) * f) & ~1
        self.out_h = int((np.tan(el1) - np.tan(el0)) * f) & ~1
        self._float_maps, masks = [], []
        src_ones = np.ones((lens.height, lens.width), np.uint8) * 255
        for R_cam in (R_wl, R_wr):
            mx, my = build_cylindrical_maps(lens, R_cam, self.out_w, self.out_h,
                                            yaw0, yaw1, el0, el1)
            self._float_maps.append((mx, my))
            masks.append(cv2.remap(src_ones, mx, my, cv2.INTER_NEAREST, borderValue=0))
        self.maps = [cv2.convertMaps(mx, my, cv2.CV_16SC2) for mx, my in self._float_maps]
        seam_feather = max(2, int(feather_px * scale))
        self.w_l = seam_weights(masks[0], masks[1], yaw0, yaw1,
                                (yaw0 + yaw1) / 2, seam_feather)
        self.w_r = (1.0 - self.w_l).astype(np.float32)
        self.gain_l = (1.0, 1.0, 1.0, 1.0)
        self.gain_r = (1.0, 1.0, 1.0, 1.0)
        self._masks = masks

    def warp(self, img, side: int):
        return cv2.remap(img, *self.maps[side], interpolation=cv2.INTER_LINEAR)

    def set_gains_from(self, img_l, img_r):
        g_l, g_r = compute_gains(self.warp(img_l, 0), self.warp(img_r, 1),
                                 self._masks[0], self._masks[1])
        self.gain_l = tuple(g_l) + (1.0,)
        self.gain_r = tuple(g_r) + (1.0,)

    def render(self, img_l, img_r):
        warp_l = cv2.multiply(self.warp(img_l, 0), self.gain_l)
        warp_r = cv2.multiply(self.warp(img_r, 1), self.gain_r)
        return cv2.blendLinear(warp_l, warp_r, self.w_l, self.w_r)

    # ---------------------------------------------------------- 심 Y 정렬

    def _measure_seam_dy(self, img_l, img_r, band=90, margin=28, tile_h=120):
        """심 밴드에서 R 이 L 대비 세로로 얼마나 어긋났는지 (행별 dy, px) 측정.

        L 밴드를 템플릿으로 R 밴드(세로 여유 margin)에서 매칭. 반환 dy>0 은
        같은 내용이 R 에서 dy 픽셀 아래에 있음을 뜻한다.
        """
        xc = self.out_w // 2
        gl = cv2.cvtColor(self.warp(img_l, 0), cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(self.warp(img_r, 1), cv2.COLOR_BGR2GRAY)
        rows, dys, wts = [], [], []
        for y0 in range(0, self.out_h - tile_h, tile_h):
            y1 = y0 + tile_h
            tmpl = gl[y0 + margin : y1 - margin, xc - band : xc + band]
            if tmpl.std() < 4:   # 무늬 없는 잔디뿐이면 신뢰 불가
                continue
            region = gr[max(0, y0) : min(self.out_h, y1), xc - band : xc + band]
            if region.shape[0] < tmpl.shape[0] + 4:
                continue
            res = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            if score < 0.35:
                continue
            dys.append(float(loc[1] - margin))
            rows.append((y0 + y1) / 2)
            wts.append(float(score))
        return np.array(rows), np.array(dys), np.array(wts)

    def refine_seam(self, img_l, img_r, taper_px=360, log=print):
        """심 밴드의 세로 어긋남을 실측해 R remap 을 심 주변에서만 국소 보정.

        시차 기인 오프셋이라 회전으로는 제거 불가 — R 소스 샘플 위치를
        행별 dy 만큼 이동시키되, 심에서 멀어질수록 0 으로 테이퍼.
        """
        rows, dys, wts = self._measure_seam_dy(img_l, img_r)
        if len(rows) < 3:
            log("[seam-refine] 측정 타일 부족 — 생략")
            return 0.0
        # 행에 대한 2차 다항 가중 피팅 (완만한 시차 프로파일)
        coef = np.polyfit(rows, dys, 2, w=wts)
        dy_fit = np.polyval(coef, np.arange(self.out_h)).astype(np.float32)
        dy_fit = np.clip(dy_fit, -30 * self.scale - 5, 30 * self.scale + 5)
        rms0 = float(np.sqrt(np.average(dys**2, weights=wts)))

        taper = np.clip(1.0 - np.abs(np.arange(self.out_w) - self.out_w // 2)
                        / max(1, int(taper_px * self.scale)), 0.0, 1.0).astype(np.float32)
        # 출력 (x,y) 에서 R 소스를 (x, y + dy·taper) 의 기존 맵 값으로 샘플
        gx, gy = np.meshgrid(np.arange(self.out_w, dtype=np.float32),
                             np.arange(self.out_h, dtype=np.float32))
        gy = gy + dy_fit[:, None] * taper[None, :]
        mx, my = self._float_maps[1]
        mx2 = cv2.remap(mx, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        my2 = cv2.remap(my, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        self._float_maps[1] = (mx2, my2)
        self.maps[1] = cv2.convertMaps(mx2, my2, cv2.CV_16SC2)

        # 검증 재측정 — 악화되면 롤백 (부호/피팅 안전장치)
        rows2, dys2, wts2 = self._measure_seam_dy(img_l, img_r)
        rms1 = float(np.sqrt(np.average(dys2**2, weights=wts2))) if len(rows2) >= 3 else rms0
        if rms1 > rms0:
            self._float_maps[1] = (mx, my)
            self.maps[1] = cv2.convertMaps(mx, my, cv2.CV_16SC2)
            log(f"[seam-refine] 개선 없음 (rms {rms0:.1f}→{rms1:.1f}px) — 롤백")
            return 0.0
        log(f"[seam-refine] 심 세로 어긋남 rms {rms0:.1f}px → {rms1:.1f}px")
        return rms0 - rms1
