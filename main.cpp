#include <QCoreApplication>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <QtWidgets/QMainWindow>

class MW : public QMainWindow
{
public:
    MW(QWidget* parent) : QMainWindow(parent)
    {
        cv::Mat image = cv::imread("C:\\Code\\PosterAugument\\Assets\\JollyRoger.jpg", 1);
        cv::namedWindow("MyView");
        cv::imshow("MyView", image);
    }
};

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    auto mw = new MW(nullptr);
    mw->show();


    return a.exec();
}
