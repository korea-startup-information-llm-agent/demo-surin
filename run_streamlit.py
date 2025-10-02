#!/usr/bin/env python3
"""
StartMate Streamlit 앱 실행 스크립트
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """필요한 패키지가 설치되어 있는지 확인"""
    try:
        import streamlit
        import langchain
        import langgraph
        print("✅ 필요한 패키지들이 설치되어 있습니다.")
        return True
    except ImportError as e:
        print(f"❌ 필요한 패키지가 설치되지 않았습니다: {e}")
        print("다음 명령어로 설치하세요:")
        print("pip install -r requirements_streamlit.txt")
        return False

def check_env_file():
    """환경 변수 파일이 있는지 확인"""
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env 파일이 존재합니다.")
        return True
    else:
        print("⚠️ .env 파일이 없습니다.")
        print("필요한 환경 변수들을 설정해주세요:")
        print("- UPSTAGE_API_KEY")
        print("- TAVILY_API_KEY") 
        print("- QDRANT_URL")
        print("- QDRANT_API_KEY")
        print("- EMBED_URL")
        return False

def run_streamlit():
    """Streamlit 앱 실행"""
    try:
        print("🚀 StartMate Streamlit 앱을 시작합니다...")
        print("브라우저에서 http://localhost:8501 로 접속하세요.")
        
        # Streamlit 앱 실행
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 앱이 종료되었습니다.")
    except Exception as e:
        print(f"❌ 앱 실행 중 오류가 발생했습니다: {e}")

def main():
    """메인 함수"""
    print("🤖 StartMate - 지식재산권 챗봇")
    print("=" * 50)
    
    # 환경 확인
    if not check_requirements():
        return
    
    if not check_env_file():
        print("⚠️ 환경 변수 설정 후 다시 실행해주세요.")
        return
    
    # 앱 실행
    run_streamlit()

if __name__ == "__main__":
    main()
