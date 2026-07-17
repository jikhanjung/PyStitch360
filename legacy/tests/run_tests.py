"""
테스트 실행 스크립트
모든 유닛 테스트를 실행하고 결과를 리포트
"""

import unittest
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_all_tests():
    """모든 테스트 실행"""
    # 테스트 디스커버리
    loader = unittest.TestLoader()
    test_suite = loader.discover(
        start_dir=Path(__file__).parent,
        pattern='test_*.py'
    )
    
    # 테스트 실행
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    print("=" * 70)
    print("PyStitch360 유닛 테스트 실행")
    print("=" * 70)
    
    result = runner.run(test_suite)
    
    # 결과 요약
    print("\n" + "=" * 70)
    print("테스트 결과 요약")
    print("=" * 70)
    print(f"실행된 테스트: {result.testsRun}")
    print(f"실패: {len(result.failures)}")
    print(f"오류: {len(result.errors)}")
    print(f"건너뜀: {len(result.skipped)}")
    
    if result.failures:
        print("\n실패한 테스트:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\n오류가 발생한 테스트:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    # 성공 여부 반환
    return result.wasSuccessful()


def run_specific_test(test_module):
    """특정 테스트 모듈 실행"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_module)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # 특정 테스트 모듈 실행
        test_module = sys.argv[1]
        success = run_specific_test(test_module)
    else:
        # 모든 테스트 실행
        success = run_all_tests()
    
    # 종료 코드 설정
    sys.exit(0 if success else 1)