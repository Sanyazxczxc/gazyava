"""Запуск дашборда кнопкой Run в PyCharm: python run.py

Дашборд слушает только этот компьютер (127.0.0.1), чтобы его не открыли из локальной сети: в нём
настройки Telegram и ключ OpenRouter. Открыть доступ для других устройств: DEN_HOST=0.0.0.0.
"""
import os
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
    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py")),
                "--server.address", os.environ.get("DEN_HOST", "127.0.0.1")]
    sys.exit(stcli.main())
