"""Запуск уведомлений в Telegram кнопкой Run в PyCharm: python run_alerts.py [--once|--test|--chat-id]"""
import sys

from den.alerts import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
