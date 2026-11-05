"""Запуск дашборда кнопкой Run в PyCharm: python run.py"""
import sys
from pathlib import Path

try:
    from streamlit.web import cli as stcli
except ImportError:
    sys.exit(
        "Streamlit не установлен в этот интерпретатор:\n  "
        + sys.executable
        + "\nВыполните: python -m pip install -r requirements.txt"
    )

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py"))]
    sys.exit(stcli.main())
