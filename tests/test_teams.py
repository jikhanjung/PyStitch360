"""classify_teams 시드 전파 — 팀 수정 격리 회귀 (devlog 031 버그)."""
from conftest import make_team_analysis

from pystitch.core.ptz import classify_teams


def test_team_fix_does_not_drag_others():
    """한 선수 팀 수정이 다른 선수(심판 포함)를 끌고 가면 안 된다.

    그늘진 선수 7의 색은 두 팀 중간 — 7을 상대 팀으로 수정했을 때
    그 애매한 색이 팀 센터가 되면 어두운 옷 심판(21)이 끌려오던 버그.
    """
    ana = make_team_analysis()
    base = classify_teams(ana)
    blue_team = base[11]
    fixed = classify_teams(ana, roles={7: blue_team})
    assert fixed[7] == blue_team                       # 수정 대상은 반영
    changed = [t for t in base if t != 7 and fixed[t] != base[t]]
    assert changed == []                               # 다른 선수 불변


def test_referee_seed_still_propagates():
    """GK/심판(역할 ≥3) 색 전파는 유지 — 같은 색 다른 조각까지 잡는다."""
    ana = make_team_analysis()
    ana["players"] = [r + [[2600, 705, 40, 120, 22, 61, 148, 62]]
                      for r in ana["players"]]
    out = classify_teams(ana, roles={21: 5})
    assert out[21] == 5 and out[22] == 5


def test_gk_temporal_exclusivity():
    """GK 단일성: 팀당 GK 한 명 — 같은 GK 역할이 시간상 겹치는 여러
    트랙릿에 전파되면 안 된다. 비겹침 조각(ID 갈라짐)은 전파 유지."""
    gk = (30, 220, 200)                              # 노란 GK 유니폼
    players = []
    for si in range(100):
        rows = []
        for tid in range(1, 7):                       # 빨강 팀
            rows.append([100 * tid, 500, 40, 120, tid,
                         2 + tid % 3, 200 - tid, 170 + tid])
        for tid in range(11, 17):                     # 파랑 팀
            rows.append([100 * tid, 520, 40, 120, tid,
                         120 + tid % 3, 190 - tid % 5, 160 + tid % 7])
        if si < 50:                                   # GK 시드 트랙릿
            rows.append([300, 700, 40, 120, 31, *gk])
        if 25 <= si < 80:                             # 같은 색, 시드와 겹침
            rows.append([3000, 700, 40, 120, 32, *gk])
        if si >= 60:                                  # 같은 색, 시드와 비겹침
            rows.append([320, 700, 40, 120, 33, *gk])
        players.append(rows)
    ana = {"fps": 30.0, "frames": [si * 3 for si in range(100)],
           "players": players}
    out = classify_teams(ana, roles={31: 3})
    assert out[31] == 3                               # 시드 확정
    assert out[33] == 3                               # 비겹침 조각 전파 유지
    assert out[32] != 3                               # 겹침 → 전파 취소
