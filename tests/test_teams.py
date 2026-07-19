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
