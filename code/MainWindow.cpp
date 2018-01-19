#include "MainWindow.h"

MainWindow::MainWindow(QWidget* _parent)
{    
    cv::Mat rawMat = cv::imread(sourceDir.path().toStdString()+"/read_board_test.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Size rawSize = cv::Size(rawMat.cols, rawMat.rows);
    std::cout << "image size: " << rawSize << std::endl;
    int dim = std::sqrt(rawMat.cols*rawMat.cols + rawMat.rows*rawMat.rows);

    double scl = 1500.0/dim;  //want around x pixels for board diagonal? like 1200x900 ish
    std::cout << "rescale factor: " << scl << std::endl;
    cv::Mat sclMat;
    cv::resize(rawMat, sclMat, cv::Size(), scl, scl );

    //int dimBgSigma = dim/50;
    //dimBgSize += dimBgSize%2+1;  //make sure it's odd - size for blur must be odd

    //cv::GaussianBlur(rawMat, rawMat, cv::Size(5,5), 3, 3);
    cv::GaussianBlur(sclMat, sclMat, cv::Size(3,3), 1, 1);
    //cv::imshow("board pic", sclMat);

    cv::Mat bgMat;
    cv::GaussianBlur(sclMat,bgMat,cv::Size(),20,20); //cv::GaussianBlur(rawMat,bgMat,cv::Size(),55,55);
    //cv::GaussianBlur(rawMat,bgMat,cv::Size(),dimBgSigma,dimBgSigma); //cv::GaussianBlur(rawMat,bgMat,cv::Size(),55,55);
    //cv::imshow("bg", bgMat);

    bgMat -= sclMat;
    cv::Mat fgMat = cv::Mat(bgMat.rows, bgMat.cols, CV_8U);

    //double minVal,maxVal;
    //cv::minMaxIdx(bgMat, &minVal, &maxVal, nullptr, nullptr);

    //std::cout << "min: " << minVal << "    max: " << maxVal << std::endl;
    cv::threshold(bgMat,fgMat,1,255, CV_8U);

    //cv::imshow("subtract", fgMat);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(fgMat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    //get contours, one of which should hopefully be the sudoku, possibly many many extraneous contours, too
    cv::Mat contourMat;
    cv::cvtColor(sclMat, contourMat, cv::COLOR_GRAY2BGR );

    //look for our sudoku. It should be roughly a square (possibly rotated and slightly warped). should also be fairly large, hopefull one of larger rects
    std::cout << "contours: " << contours.size() << std::endl;
    //int contNo = 0;
    //for (std::vector<cv::Point> cont: contours)
    double bestFitVal = 0;
    size_t bestContIdx = 0;
    for (size_t contNo=0; contNo<contours.size(); ++contNo)
    {
        cv::Moments moms = cv::moments(contours[contNo]);
        if (moms.m00 < 500)       //board needs to be pretty big to read the numbers. weed out obvious non-boards
            continue;

        //ideally rotRect.area ~= m00
        //cv::Rect bRect = cv::boundingRect(cont);
        cv::RotatedRect rotRect = cv::minAreaRect(contours[contNo]);
        cv::Rect brRect = rotRect.boundingRect();
        //int rArea = bRect.width * bRect.height;
        double rArea = rotRect.size.area();

        std::cout << "c#: " << contNo << "\t";
        //std::cout << "rect: " << bRect << "\t";
        std::cout << "area: " << int(moms.m00) << "  \t";
        std::cout << "rotA: " << int(rArea) << "\t";
        std::cout << "rotR: " << rotRect.angle << "<  " << rotRect.center << "\t";

        //std::cout << "rArea: " << rArea << "\t";

        double areaDiff = 1 - (std::abs(moms.m00 - rArea) / rArea );
        double aspectDiff = 1 - (std::abs(brRect.width - brRect.height) / std::max(brRect.width, brRect.height));
        double fitVal = std::sqrt(rArea) * areaDiff * areaDiff * aspectDiff;  //some value I think is reasonable to determine the fitness of a contour as a sudoku box. certainly not machine-learning/AI level of assessment, I know
        std::cout << "fit: " << int(fitVal) << std::endl;

        if (fitVal > bestFitVal)
        {
            bestContIdx = contNo;
            bestFitVal = fitVal;
        }

        cv::drawContours(contourMat, contours, contNo, cv::Scalar(255,255,255),1, cv::LINE_8);
        //++contNo;
    }

    cv::RotatedRect bestRotRect = cv::minAreaRect(contours[bestContIdx]);
    cv::Point2f rotRectPts[4];
    bestRotRect.points(rotRectPts);
    for (size_t ptNo=0; ptNo<4; ++ptNo)
    {
        //std::cout << "pt: " << rotRectPts[ptNo] << std::endl;
        cv::line(contourMat, rotRectPts[ptNo], rotRectPts[(ptNo+1)%4], cv::Scalar(0,0,255), 2);
    }

    //cv::Rect bestRect = cv::boundingRect(contours[bestContIdx]);
    //cv::rectangle(contourMat, bestRect, cv::Scalar(0,255,0), 2);

    cv::drawContours(contourMat, contours, bestContIdx, cv::Scalar(255,0,0), 2);

    std::vector<cv::Point2f> approxCont;
    double epsilon = std::sqrt(bestRotRect.boundingRect().area())/5;  // ~1/20 perimeter? am i doing this right? On test image there's quite a large range that gives the expected 4pts, like 1/2th perim to 1/200th perim
    cv::approxPolyDP(contours[bestContIdx], approxCont, epsilon, true);
    size_t aprxCnt = approxCont.size();
    std::cout << "\n approx: " << aprxCnt << std::endl;
    for (cv::Point2f& pt : approxCont)
        std::cout << pt << "\t";

    std::cout << std::endl;
    for (size_t ptNo=0; ptNo<aprxCnt; ++ptNo)
    {
        //std::cout << "pt: " << rotRectPts[ptNo] << std::endl;
        cv::line(contourMat, approxCont[ptNo], approxCont[(ptNo+1)%aprxCnt], cv::Scalar(0,255,0), 2);
    }

    std::cout << "homography" << std::endl;
    std::vector<cv::Point2f> srcPts = {cv::Point2f(0,0), cv::Point2f(0,9), cv::Point2f(9,9), cv::Point2f(9,0)};
    //cv::Mat Q;
    cv::Mat Q = cv::findHomography(srcPts, approxCont); //####
    std::cout << Q << std::endl;

    //std::vector<std::vector<cv::Point2f>> inputCenterPts(9,std::vector<cv::Point2f>(9) );
    std::vector<cv::Point2f> inputCenterPts(81);
    for (size_t row=0; row<9; ++row)
    {
        for (size_t col=0; col<9; ++col)
        {
            //inputCenterPts[row][col] = cv::Point2f(col+.5, row+.5);
            inputCenterPts[9*row+col] = cv::Point2f(col+.5, row+.5);
        }
    }

    //std::vector<std::vector<cv::Point2f>> outputCenterPts(9,std::vector<cv::Point2f>(9) );
    std::vector<cv::Point2f> outputCenterPts(81);

    std::cout << "perspective tranform" << std::endl;
    cv::Mat Qtrans;
    cv::transpose(Q,Qtrans);
    cv::perspectiveTransform(inputCenterPts, outputCenterPts,Q);
    //cv::transform(inputCenterPts, outputCenterPts, Q);

    for (size_t row=0; row<9; ++row)
    {
        for (size_t col=0; col<9; ++col)
        {
            //std::cout << outputCenterPts[9*row+col] << "\t";
            //std::cout << outputCenterPts[row][col] << "\t";
            cv::circle(contourMat, outputCenterPts[9*row+col], 3, cv::Scalar(0,255,255), -1);
        }
        std::cout << std::endl;
    }

    //test warp visually ### un-warps the input image so the sudoku board lies in a nice square mat
    //where the mat/board boundaries are one and the same (because input image will not be perfectly in-plane and cropepd to board)
    int cellDim = 28;
    int tranSize = 9*cellDim;  //252=28, 216=24, 576=64
    std::vector<cv::Point2f> mapPts = {cv::Point2f(0,0), cv::Point2f(0,tranSize), cv::Point2f(tranSize,tranSize), cv::Point2f(tranSize,0)};
    cv::Mat H = cv::findHomography(approxCont, mapPts );
    cv::Mat warpMat, warpInput;
    std::cout << H << std::endl;
    //sclMat.copyTo(warpInput);
    //warpInput = sclMat( cv::boundingRect(contours[bestContIdx]) );

    //bgMat?
    cv::warpPerspective(fgMat, warpMat, H, cv::Size(tranSize,tranSize) ); // warpInput.size() );  //warpInput
    //cv::imshow("pre-warp", warpInput);
    cv::imshow("warp", warpMat);

//    for (size_t row=0; row<9; ++row)
//    {
//        for (size_t col=0; col<9; ++col)
//        {
//            cv::Point2f sqrCenter(col+.5, row+.5);
//            cv::Point2f transfPt;
//            cv::perspectiveTransform()
//            cv::circle()

//        }
//    }

    //std::cout << Q << std::endl;
    //i think from here we can look for contours centered moms.(m01/m00, m10/m00) closest to the cetnerpoint of each square that we can immediately calculate now...

    //cv::drawContours(contourMat, contours, -1, cv::Scalar(0,255,0),1, cv::LINE_8);
    cv::imshow("contours", contourMat);
    cv::waitKey(0);

    digReader.Train_Recognizer("", "");  //this might take some time I'd assume?

    //size_t testRow=0, testCol=2;
    for (size_t testRow=0; testRow<9; ++testRow)
    {
        for (size_t testCol=0; testCol<9; ++testCol)
        {
            cv::Rect testDetectRect(cellDim*testCol + 4, cellDim*testRow + 4, 20, 20);  //### if change cellDim change this +offset!
            cv::Mat testDetectMat = warpMat(testDetectRect);
            cv::Mat testDetectFloat;
            if (cv::mean(testDetectMat)[0] < 16)
                continue;

            //this bit not needed if using fgImg as base for earp (fgImg is threshoded so 0 or 255, bgImg, is not normlized and would need this bit);
            //double minVal,maxVal,sclToMax;
            //cv::minMaxIdx(testDetectMat, &minVal, &maxVal, nullptr, nullptr);
            //sclToMax = 255/maxVal;
            //testDetectMat *= sclToMax;

            testDetectMat.convertTo(testDetectFloat, CV_32FC1);
            std::cout << "test (" << testRow << "," << testCol << ")   " << digReader.Classify_Image( testDetectFloat.reshape(1,1) ) << std::endl;
            cv::imshow("test detect image", testDetectMat);
            cv::waitKey(500);
        }
    }

    //digReader.train()
    //Create_Legend_Window();
}
