#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QWidget>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QComboBox>
#include <QLabel>
#include <QPushButton>
#include <QDir>
#include <QFile>
#include <QString>
#include <QSpinBox>
#include <QTextStream>
#include <QIODevice>

#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/calib3d.hpp"

class MainWindow : public QWidget
{
    Q_OBJECT
private:
    QPushButton* pushButtonSolve, *pushButtonLoad, *pushButtonSlowSolve, *pushButtonPauseSolve;
    QComboBox* comboBoxFileSelect;
    QLabel* labelFileLabel, *labelPuzzleNo;
    QSpinBox* spinBoxPuzzleNo;
    QGridLayout* gLayoutMain;
    QHBoxLayout* hLayoutLoad;
    QVBoxLayout* vLayoutMain;
    QDir sourceDir;
    QWidget* windowLegend;

public:
    MainWindow(QWidget* _parent = 0);

signals:


private slots:

};

#endif
