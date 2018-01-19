#include "MainWindow.h"

#include <QApplication>

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    SudokuWindow sWindow;
    sWindow.show();
    return app.exec();
}
