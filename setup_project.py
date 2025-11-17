"""
프로젝트 초기 설정 스크립트
폴더 구조를 자동으로 생성합니다.
"""

import os

def create_project_structure():
    """프로젝트 폴더 구조 생성"""
    
    # 생성할 폴더 목록
    folders = [
        'data/raw/subway',
        'data/raw/weather',
        'data/raw/congestion',
        'data/processed',
        'data/external',
        'notebooks',
        'src',
        'models',
        'app/templates',
        'app/static/css',
        'app/static/js',
        'app/static/images',
        'tests',
        'scripts'
    ]
    
    print("="*70)
    print("🚀 프로젝트 초기 설정 시작")
    print("="*70)
    print()
    
    # 폴더 생성
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✓ {folder}/")
    
    # .gitkeep 파일 생성 (빈 폴더도 Git에 포함)
    gitkeep_folders = [
        'data/raw/subway',
        'data/raw/weather',
        'data/raw/congestion',
        'data/processed',
        'data/external',
        'models'
    ]
    
    print()
    print("📝 .gitkeep 파일 생성 중...")
    for folder in gitkeep_folders:
        gitkeep_path = os.path.join(folder, '.gitkeep')
        with open(gitkeep_path, 'w') as f:
            pass
        print(f"✓ {gitkeep_path}")
    
    # __init__.py 파일 생성
    print()
    print("📝 __init__.py 파일 생성 중...")
    init_folders = ['src', 'app', 'tests']
    for folder in init_folders:
        init_path = os.path.join(folder, '__init__.py')
        with open(init_path, 'w') as f:
            f.write(f'"""{folder} package"""\n')
        print(f"✓ {init_path}")
    
    print()
    print("="*70)
    print("✅ 프로젝트 폴더 구조 생성 완료!")
    print("="*70)
    print()
    print("다음 단계:")
    print("1. .env.template을 .env로 복사")
    print("2. .env 파일에 API 키 입력")
    print("3. pip install -r requirements.txt")
    print()

if __name__ == "__main__":
    create_project_structure()
