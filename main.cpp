#include <QCoreApplication>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    cv::Mat image = cv::imread("C:\\Code\\PosterAugument\\Assets\\JollyRoger.jpg");
    cv::namedWindow("MyView");
    cv::imshow("MyView", image);

    return a.exec();
}
