"""2패스 PTZ 계획(build_plan) 유닛 테스트 — ultralytics 불필요 (순수 numpy)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.ptz import build_plan  # noqa: E402

PANO_W, PANO_H, FPS = 5906, 1680, 30.0


def _analysis(frames, balls, players=None):
    n = len(frames)
    return {"fps": FPS, "total_frames": int(frames[-1]) + 1,
            "frames": list(frames),
            "balls": balls,
            "players": players if players is not None else [[] for _ in range(n)]}


def test_follows_ball_and_fills_gap():
    """직선 이동하는 공 + 1초 미검출 갭 → 궤적 추종, 갭 보간, 줌 1배."""
    frames = list(range(0, 900, 3))
    balls = []
    for f in frames:
        x = 1500 + 3.0 * f          # 완만한 이동 (90px/s)
        balls.append(None if 300 <= f < 330 else [x, 900.0, 0.5])
    plan = build_plan(_analysis(frames, balls), PANO_W, PANO_H, log=None)
    # 갭 구간 포함 전 구간에서 공 근처 (스무딩 지연 고려 넉넉히)
    err = np.abs(plan["cx"][150:850] - (1500 + 3.0 * np.arange(150, 850)))
    assert err.max() < 120
    # y=900(중거리) 공: 근경 가변 줌 반영폭 안에서 안정적으로 유지
    assert np.all(plan["crop_w"][150:850] < 1920 * 1.35)
    assert plan["crop_w"][150:850].std() < 40


def test_rejects_teleporting_low_conf_ball():
    """저신뢰 순간이동 검출(장외 공 등)은 게이팅으로 무시."""
    frames = list(range(0, 300, 3))
    balls = [[2000.0, 900.0, 0.5] for _ in frames]
    balls[50] = [5500.0, 600.0, 0.3]   # 3500px 순간이동, conf 0.3
    plan = build_plan(_analysis(frames, balls), PANO_W, PANO_H, log=None)
    assert np.abs(plan["cx"] - 2000).max() < 60


def test_zoom_out_when_no_ball():
    """공 부재 + 선수 산개 → 선수 분포를 덮도록 줌아웃, 중심은 군집 근처."""
    frames = list(range(0, 900, 3))
    rng = np.random.default_rng(3)
    players = [[[float(x), 900.0] for x in rng.uniform(2200, 4800, 20)]
               for _ in frames]
    plan = build_plan(_analysis(frames, [None] * len(frames), players),
                      PANO_W, PANO_H, log=None)
    mid = slice(200, 700)
    # 20명 표본의 p10~p90 기대폭 ~1900px + 마진 520px
    assert np.all(plan["crop_w"][mid] > 2200)
    assert np.abs(plan["cx"][mid] - 3500).max() < 400  # 군집 중심 부근


def test_smoothness_frame_step_bounded():
    """노이즈 낀 검출에도 프레임당 이동량이 작아야 함 (흔들림 방지)."""
    frames = list(range(0, 1800, 3))
    rng = np.random.default_rng(5)
    balls = [[2800.0 + rng.normal(0, 80), 900.0 + rng.normal(0, 40), 0.45]
             for _ in frames]
    plan = build_plan(_analysis(frames, balls), PANO_W, PANO_H, log=None)
    step = np.abs(np.diff(plan["cx"][60:-60]))
    assert step.max() < 4.0            # 정지 상황: 프레임당 4px 미만
    assert np.abs(plan["cx"][60:-60] - 2800).max() < 80


def test_fast_ball_engages_fast_follow():
    """공이 빠르게 관통(2500px/2s)하면 빠른 추종으로 전환되어 크게 뒤처지지 않음."""
    frames = list(range(0, 1800, 3))
    balls = []
    for f in frames:
        if f < 600:
            x = 1200.0
        elif f < 660:
            x = 1200.0 + (f - 600) / 60 * 2500   # 2초에 2500px
        else:
            x = 3700.0
        balls.append([x, 900.0, 0.5])
    plan = build_plan(_analysis(frames, balls), PANO_W, PANO_H, log=None)
    # 이동 완료 1.5초 뒤에는 목표에 수렴해 있어야 함
    assert abs(plan["cx"][660 + 45] - 3700) < 250
    # 크롭이 파노라마 밖을 요구하지 않음
    half = plan["crop_w"] / 2
    assert np.all(plan["cx"].clip(half, PANO_W - half) - plan["cx"] < 1e-6 + half)


def test_static_isolated_decoy_rejected():
    """장시간 정지 + 선수들로부터 고립된 '공'(낙엽 등)은 기각 → 줌아웃 유지."""
    frames = list(range(0, 900, 3))
    balls = [[4900.0, 1350.0, 0.6] for _ in frames]      # 완전 정지, 높은 conf
    players = [[[float(x), 800.0] for x in np.linspace(1200, 4100, 8)]
               for _ in frames]                           # 전원 700px+ 거리, 넓게 산개
    plan = build_plan(_analysis(frames, balls, players), PANO_W, PANO_H, log=None)
    mid = slice(200, 700)
    assert np.abs(plan["cx"][mid] - 4900).min() > 1200   # 미끼를 안 따라감
    assert np.all(plan["crop_w"][mid] > 1920 * 1.05)     # 선수 산개 커버 줌아웃

    # 같은 정지 공이라도 선수가 근처에 있으면(세트피스) 정상 추적
    players2 = [[[4850.0, 1300.0], [4700.0, 1250.0], [3000.0, 800.0]]
                for _ in frames]
    plan2 = build_plan(_analysis(frames, balls, players2), PANO_W, PANO_H, log=None)
    assert np.abs(plan2["cx"][mid] - 4900).max() < 100


def test_keyframes_override_auto():
    """키프레임은 자동 검출을 덮어쓰고(±1.5s 억제), 멀리서는 자동 추종 유지."""
    frames = list(range(0, 1800, 3))
    balls = [[2000.0, 900.0, 0.5] for _ in frames]          # 자동: 내내 x=2000
    plan = build_plan(_analysis(frames, balls), PANO_W, PANO_H,
                      keyframes=[(900, 4000.0, 1000.0)], log=None)
    assert abs(plan["cx"][900] - 4000) < 350        # 키프레임 지점: 수동 위치
    assert abs(plan["cx"][150] - 2000) < 50         # 먼 곳: 자동 유지
    assert abs(plan["cx"][1700] - 2000) < 50


def test_keyframe_pair_bridges_detection_gap():
    """자동 검출이 전혀 없는 구간도 인접 키프레임(<=8s)끼리 직접 보간."""
    frames = list(range(0, 2700, 3))
    balls = [None] * len(frames)                    # 자동 검출 전무
    kfs = [(300, 1500.0, 900.0), (480, 3000.0, 900.0)]   # 6초 간격
    plan = build_plan(_analysis(frames, balls), PANO_W, PANO_H,
                      keyframes=kfs, log=None)
    mid = plan["cx"][390]                           # 중간 지점 ≈ 평균
    assert abs(mid - 2250) < 300
    assert np.all(plan["crop_w"][360:420] < 1920 * 1.25)  # 구간 중앙: 줌인 유지


def test_keyframe_zoom_override():
    """키프레임 4번째 값(crop_w)이 그 시점 크롭 폭을 좌우한다 (수동 줌)."""
    frames = list(range(0, 900, 3))
    balls = [[2900.0, 900.0, 0.5] for _ in frames]
    base = build_plan(_analysis(frames, balls), PANO_W, PANO_H,
                      keyframes=[(450, 2900.0, 900.0)], log=None)
    tight = build_plan(_analysis(frames, balls), PANO_W, PANO_H,
                       keyframes=[(450, 2900.0, 900.0, 800.0)], log=None)
    assert tight["crop_w"][450] < base["crop_w"][450] - 100   # 더 타이트
    assert abs(tight["crop_w"][450] - 800.0) < 200            # 앵커가 근처로
    # 3-요소(줌 없음) 키프레임은 기존과 동일하게 동작 (하위 호환)
    assert abs(base["crop_w"][450] - tight["crop_w"][450]) > 100


def test_wide_mode_fixed_zoom_gentle_pan():
    """와이드 모드: 크롭 폭 고정(최대), 가로 팬은 완만."""
    frames = list(range(0, 2700, 3))
    rng = np.random.default_rng(9)
    balls = [[2000.0 + 1.0 * f + rng.normal(0, 60), 900.0, 0.5]
             for f in frames]                       # 완만 이동 + 노이즈
    plan = build_plan(_analysis(frames, balls), PANO_W, PANO_H,
                      out_w=2560, out_h=1080, wide=True, sigma_slow=3.0,
                      fast_err_px=800.0, log=None)
    max_w = min(PANO_W, (PANO_H - 160) * 2560 / 1080)
    assert np.allclose(plan["crop_w"], max_w, atol=1.0)   # 줌 변동 없음
    step = np.abs(np.diff(plan["cx"][90:-90]))
    assert step.max() < 3.0                               # 완만한 팬


def test_ignore_ranges_kill_track():
    """사용자 무시 구간과 겹치는 트랙은 통째로 기각."""
    from pystitch.core.ptz import accept_ball_tracks
    frames = list(range(0, 900, 3))
    balls = [[2000.0 + f, 900.0, 0.5] for f in frames]
    a = _analysis(frames, balls)
    _, ball, spans = accept_ball_tracks(a)
    assert len(spans) == 1 and not np.isnan(ball[50, 0])
    _, ball2, spans2 = accept_ball_tracks(a, ignore_ranges=[(400, 500)])
    assert spans2 == [] and np.all(np.isnan(ball2[:, 0]))   # 한 트랙 전체 기각


def test_force_ranges_promote_rejected_track():
    """승격(무시의 반대): 자동이 기각한 정지·고립 트랙도 되살린다. 무시 우선."""
    from pystitch.core.ptz import accept_ball_tracks
    frames = list(range(0, 900, 3))
    balls = [[4900.0, 1350.0, 0.6] for _ in frames]        # 정지+고립 → 자동 기각
    players = [[[float(x), 800.0] for x in np.linspace(1200, 4100, 8)]
               for _ in frames]
    a = _analysis(frames, balls, players)
    _, _, spans0 = accept_ball_tracks(a)
    assert spans0 == []                                    # 기본은 기각
    _, ball1, spans1 = accept_ball_tracks(
        a, force_ranges=[(450, 4900.0, 1350.0)])           # 그 자리 승격
    assert len(spans1) == 1 and not np.isnan(ball1[50, 0])
    _, _, spans2 = accept_ball_tracks(                      # 무시가 승격보다 우선
        a, force_ranges=[(450, 4900.0, 1350.0)],
        ignore_ranges=[(0, 897, 4900.0, 1350.0)])
    assert spans2 == []


def test_near_ball_widens_crop():
    """근경(화면 아래) 공은 크롭을 넓게, 원경 공은 타이트하게."""
    frames = list(range(0, 900, 3))
    far = build_plan(_analysis(frames, [[2800.0, 550.0, 0.5] for _ in frames]),
                     PANO_W, PANO_H, log=None)
    near = build_plan(_analysis(frames, [[2800.0, 1600.0, 0.5] for _ in frames]),
                      PANO_W, PANO_H, log=None)
    mid = slice(200, 700)
    assert far["crop_w"][mid].mean() < 2100          # 원경: 1.1배 이하
    assert near["crop_w"][mid].mean() > 2700         # 최하단: ~1.6배


def test_player_bbox_format_compatible():
    """선수가 (cx,cy,w,h) 4열이어도 계획 로직은 중심 2열만 사용해 동일 동작."""
    frames = list(range(0, 900, 3))
    players4 = [[[float(x), 900.0, 40.0, 90.0] for x in np.linspace(2200, 4800, 20)]
                for _ in frames]
    plan = build_plan(_analysis(frames, [None] * len(frames), players4),
                      PANO_W, PANO_H, log=None)
    assert np.abs(plan["cx"][300:600] - 3500).max() < 400   # 2열 케이스와 동일


def test_classify_teams_by_kit_color():
    """유니폼 색으로 팀 2개 + 기타(심판) 분류 — ID별 다수 검출."""
    from pystitch.core.ptz import classify_teams
    frames = list(range(0, 300, 3))
    rng = np.random.default_rng(2)
    def det(tid, h, s, v):
        return [1000.0, 900.0, 40.0, 90.0, tid,
                h + rng.normal(0, 3), s + rng.normal(0, 8), v + rng.normal(0, 8)]
    players = []
    for _ in frames:
        row = []
        for tid in range(0, 8):     # 팀A: 파랑 (H~120 in OpenCV 0~180)
            row.append(det(tid, 120, 180, 150))
        for tid in range(10, 18):   # 팀B: 빨강 (H~0)
            row.append(det(tid, 3, 190, 160))
        row.append(det(30, 60, 200, 200))   # 심판: 형광 노랑-초록
        players.append(row)
    a = _analysis(frames, [None] * len(frames), players)
    teams = classify_teams(a)
    ta = {teams[t] for t in range(0, 8)}
    tb = {teams[t] for t in range(10, 18)}
    assert len(ta) == 1 and len(tb) == 1 and ta != tb   # 팀 내 일관, 팀 간 상이
    assert ta | tb == {0, 1}                             # 상위 2개 군집이 팀
    assert teams[30] == 2                                # 심판은 기타


def test_classify_teams_role_seeds_propagate():
    """GK/심판 시드 지정 → 같은 색 옷의 다른 트랙릿에 역할 전파.

    GK 는 팀당 한 명 — ID 갈라짐은 시간상 순차라, 전파 대상(21/23)은
    시드와 시간이 겹치지 않는 후반 조각으로 모델링한다 (동시 존재하는
    같은 색은 GK 단일성 휴리스틱이 걸러냄 — test_teams.py 참고).
    """
    from pystitch.core.ptz import classify_teams
    frames = list(range(0, 300, 3))
    n = len(frames)
    rng = np.random.default_rng(3)
    def det(tid, h, s, v):
        return [1000.0, 900.0, 40.0, 90.0, tid,
                h + rng.normal(0, 3), s + rng.normal(0, 8), v + rng.normal(0, 8)]
    players = []
    for si in range(n):
        row = []
        for tid in range(0, 8):
            row.append(det(tid, 120, 180, 150))    # 팀A: 파랑
        for tid in range(10, 18):
            row.append(det(tid, 3, 190, 160))      # 팀B: 빨강
        if si < n // 2:                            # GK: 전반 조각 (시드)
            row.append(det(20, 60, 200, 180))      # GK A: 초록
            row.append(det(22, 150, 190, 170))     # GK B: 보라
        else:                                      # 후반 조각 (전파 대상)
            row.append(det(21, 60, 200, 180))
            row.append(det(23, 150, 190, 170))
        row.append(det(30, 25, 210, 210))          # 심판: 노랑
        players.append(row)
    a = _analysis(frames, [None] * len(frames), players)
    # 시드: GK A=20, GK B=22, 심판=30 만 지정 — 21/23 은 색으로 전파돼야 함
    teams = classify_teams(a, roles={20: 3, 22: 4, 30: 5})
    assert teams[20] == 3 and teams[21] == 3             # 같은 초록 → GK A
    assert teams[22] == 4 and teams[23] == 4             # 같은 보라 → GK B
    assert teams[30] == 5
    ta = {teams[t] for t in range(0, 8)}
    tb = {teams[t] for t in range(10, 18)}
    assert ta | tb == {0, 1} and ta != tb                # 필드 플레이어는 그대로
    # 시드 없이 부르면 기존 0/1/2 동작 유지
    base = classify_teams(a)
    assert set(base.values()) <= {0, 1, 2}


def test_tracklet_colors_circular_hue():
    """유니폼 대표색: H 0/180 경계(빨강)에서도 중앙값이 튀지 않는다."""
    from pystitch.core.ptz import tracklet_colors
    frames = list(range(0, 60, 3))
    players = []
    for i, _ in enumerate(frames):
        h = 178.0 if i % 2 == 0 else 2.0     # 경계 양쪽을 오가는 빨강
        players.append([[1000.0, 900.0, 40.0, 90.0, 7, h, 200.0, 150.0]])
    a = _analysis(frames, [None] * len(frames), players)
    h, s, v = tracklet_colors(a)[7]
    assert h <= 5.0 or h >= 175.0                        # 빨강 근방 유지
    assert 150 <= s <= 210 and 130 <= v <= 170


def test_gapfill_targets_interpolate_gaps():
    """수락 트랙 사이 짧은 갭만 보간 목표로 — 긴 갭(플레이 중단)은 제외."""
    from pystitch.core.ptz import gapfill_targets
    frames = list(range(0, 3000, 3))
    balls = []
    for f in frames:
        if 300 <= f < 600 or 660 <= f < 900:      # 2초 갭 (660-600)
            balls.append([1000.0 + f, 800.0, 0.6])
        elif 1800 <= f < 2100:                     # 앞 트랙과 30초 갭
            balls.append([2500.0, 800.0, 0.6])
        else:
            balls.append(None)
    a = _analysis(frames, balls)
    a.update({"pano_w": PANO_W, "pano_h": PANO_H, "detect_every": 3})
    tg = gapfill_targets(a, max_gap_s=4.0)
    assert tg, "짧은 갭 목표가 있어야 함"
    sis = [t[0] for t in tg]
    # 짧은 갭(600~660 프레임 = 샘플 200~220) 안에만 목표가 생겨야 함
    assert min(sis) >= 200 and max(sis) <= 220
    # 보간 위치는 양 끝 사이에서 단조 증가 (공이 +x 로 이동 중)
    xs = [t[1] for t in tg]
    assert all(b > a_ for a_, b in zip(xs, xs[1:]))
    assert 1590 <= min(xs) and max(xs) <= 1670
    # 긴 갭(900~1800)은 목표 없음
    assert not any(300 < si < 600 for si in sis)


def test_radar_panel_aspect_and_data():
    """내보내기 레이더: 경기장 사각형이 입력 크기(100×62) 비율 그대로."""
    from pystitch.core.ptz import build_radar_data, draw_radar_panel
    frames = [0, 3]
    a = {"frames": frames, "balls": [[100.0, 900.0, 0.9], None],
         "players": [[[100.0, 900.0, 30.0, 80.0, 1]], []],
         "pano_w": PANO_W, "pano_h": PANO_H}
    radar = build_radar_data(a, {1: 0}, field_size=(100.0, 62.0),
                             palette={0: (0, 0, 255)})
    img = draw_radar_panel(radar, 0, 384)
    assert img.shape[1] == 384
    # 등방 축척: 세로/가로 = (폭+여백)/(길이+여백)
    assert abs(img.shape[0] / 384 - (62 + 8) / (100 + 8)) < 0.02
    assert len(radar["points"][0]) == 1 and radar["balls"][0] is not None
    assert radar["points"][1] == [] and radar["balls"][1] is None


def test_ground_positions_geometry():
    """지면 투영 기하: 화면 중앙 열은 X=0, 아래 행일수록 가깝고 대칭."""
    from pystitch.core.ptz import ground_positions
    W, H, h_cam = 5906, 1680, 4.0
    # 중앙 열, 세로 여러 위치의 발끝 (박스 h=0 으로 발=cy)
    cx = (W - 1) / 2                                # 픽셀 0..W-1 의 정중앙
    rows = [[cx, y, 40.0, 0.0] for y in (600.0, 1000.0, 1500.0)]
    pts = ground_positions(rows, W, H, cam_height=h_cam)
    assert len(pts) == 3
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    assert all(abs(x) < 1e-6 for x in xs)          # 중앙 열 → X=0
    assert ys[0] > ys[1] > ys[2] > 0               # 위 행일수록 멀다
    # 좌우 대칭
    pl = ground_positions([[cx - 800, 1000.0, 40.0, 0.0]], W, H, cam_height=h_cam)
    pr = ground_positions([[cx + 800, 1000.0, 40.0, 0.0]], W, H, cam_height=h_cam)
    assert abs(pl[0][0] + pr[0][0]) < 1e-6 and abs(pl[0][1] - pr[0][1]) < 1e-6
    # 수평선 근처는 제외
    assert ground_positions([[cx, 300.0, 40.0, 0.0]], W, H) == []


def test_export_training_labels():
    """무시=하드네거티브, 수락=양성, 키프레임=수동 양성 — 원본 비파괴."""
    from pystitch.core.ptz import export_training_labels
    frames = list(range(0, 600, 3))
    # 진짜 공 트랙(이동) + 300~420 구간의 정지 오인식(낙엽형, 별도 트랙)
    balls = [[5000.0, 1400.0, 0.5, 14.0, 14.0] if 300 <= f <= 420
             else [2000.0 + f, 900.0, 0.5, 14.0, 14.0] for f in frames]
    a = _analysis(frames, balls)
    import copy
    a0 = copy.deepcopy(a)
    labels = export_training_labels(a, keyframes=[(150, 3333.0, 800.0)],
                                    ignore_ranges=[(300, 420)])
    assert a == a0                                        # 분석 원본 불변
    by = {}
    for r in labels:
        by.setdefault(r["label"], []).append(r)
    assert all(300 <= r["frame"] <= 420 for r in by["not_ball"])
    assert len(by["not_ball"]) > 0 and len(by["ball"]) > 0
    assert by["ball_manual"][0]["x"] == 3333.0
    assert by["ball"][0]["w"] == 14.0                     # 박스 크기 보존


def test_same_spot_recurring_decoys_batched():
    """같은 자리에서 시간대만 다른 정적 오인식 트랙들을 일괄 수집."""
    from pystitch.core.ptz import link_ball_tracks, same_spot_spans
    frames = list(range(0, 1800, 3))
    balls = []
    for f in frames:
        if 0 <= f < 300 or 900 <= f < 1200:      # 같은 자리 정적 오인식 2회
            balls.append([4900.0, 1350.0, 0.5, 14.0, 14.0])
        elif 400 <= f < 800:                     # 진짜 공 (이동)
            balls.append([1500.0 + f, 900.0, 0.5, 14.0, 14.0])
        else:
            balls.append(None)
    linked = link_ball_tracks(_analysis(frames, balls))
    spans = same_spot_spans(linked, 0, 297)
    assert len(spans) == 2                       # 두 정적 트랙 모두
    assert spans[0][0] == 0 and spans[1][0] == 900
    # 이동하는 진짜 공 트랙은 포함되지 않음
    assert all(not (400 <= s[0] < 800) for s in spans)


def test_far_zoom_tightens_far_crop_only():
    """far_zoom: 원경 공은 더 조이고(업스케일 줌), 근경은 기존과 동일."""
    frames = list(range(0, 900, 3))
    far_balls = [[2800.0, 550.0, 0.5] for _ in frames]
    near_balls = [[2800.0, 1600.0, 0.5] for _ in frames]
    mid = slice(200, 700)
    a_far = build_plan(_analysis(frames, far_balls), PANO_W, PANO_H,
                       far_zoom=1.2, log=None)
    # y=550 은 field_top(437) 대비 depth 9% — 원경 조임과 근경 보간의 혼합값
    assert 1920 / 1.3 < a_far["crop_w"][mid].mean() < 1920 / 1.05
    base = build_plan(_analysis(frames, far_balls), PANO_W, PANO_H, log=None)
    assert a_far["crop_w"][mid].mean() < base["crop_w"][mid].mean() - 100
    a_near = build_plan(_analysis(frames, near_balls), PANO_W, PANO_H,
                        far_zoom=1.2, log=None)
    assert a_near["crop_w"][mid].mean() > 2700          # 근경 확장 유지


def test_ignored_spot_samples_removed_from_accepted_tracks():
    """무시 트랙의 '자리'에 흡수된 샘플은 수락 트랙 안이라도 제거."""
    from pystitch.core.ptz import accept_ball_tracks
    frames = list(range(0, 1200, 3))
    balls = []
    for f in frames:
        if 600 <= f < 900:
            balls.append([4900.0, 1350.0, 0.5])       # 순수 낙엽 트랙
        elif 200 <= f < 500:
            # 진짜 공이 낙엽 근처(30px)를 지남 — 트랙에 흡수될 샘플
            x = 4400.0 + (f - 200) * 2.0
            if 440 <= f < 470:
                balls.append([4910.0, 1345.0, 0.5])   # 낙엽 위치로 튄 오검출
            else:
                balls.append([x, 1000.0, 0.5])
        else:
            balls.append(None)
    a = _analysis(frames, balls)
    _, ball, spans = accept_ball_tracks(a, ignore_ranges=[(600, 897)])
    idx = np.array(a["frames"])
    # 낙엽 위치로 튄 샘플(440~470)은 수락 트랙 소속이어도 NaN
    m = (idx >= 440) & (idx < 470)
    assert np.all(np.isnan(ball[m, 0]))
    # 그 앞뒤 진짜 궤적은 유지
    keep = (idx >= 200) & (idx < 430)
    assert np.isfinite(ball[keep, 0]).mean() > 0.9


def test_multi_candidate_ball_survives_decoy_ignore():
    """미끼가 conf 로 이겨도 진짜 공이 후보로 보존되고, 위치 지정 무시는
    같은 시간대의 진짜 공 트랙을 살린다 (다중 후보의 핵심 가치)."""
    from pystitch.core.ptz import accept_ball_tracks
    frames = list(range(0, 900, 3))
    cands, balls = [], []
    for f in frames:
        leaf = [4900.0, 1350.0, 0.6, 14.0, 14.0]        # 미끼: conf 우위
        real = [1500.0 + 2.0 * f, 900.0, 0.4, 14.0, 14.0]
        cands.append([leaf, real])
        balls.append(leaf)                               # 구형 필드 = 최고 conf
    a = _analysis(frames, balls)
    a["ball_cands"] = cands
    # 위치 지정 무시 (낙엽 자리) → 낙엽 트랙만 기각
    _, ball, spans = accept_ball_tracks(
        a, ignore_ranges=[(0, 897, 4900.0, 1350.0)])
    assert len(spans) == 1
    idx = np.array(a["frames"])
    known = np.isfinite(ball[:, 0])
    assert known.mean() > 0.9                            # 진짜 공 궤적 생존
    assert np.all(np.abs(ball[known, 1] - 900.0) < 1)    # 낙엽(1350) 아님
