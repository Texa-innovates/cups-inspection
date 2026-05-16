import sys
from PyQt5.QtWidgets import QApplication

from classes.database import create_database_and_tables
from classes.main_window import MainWindow

def main():
    create_database_and_tables()
    app = QApplication(sys.argv)
    w = MainWindow()
    w.showMaximized()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
