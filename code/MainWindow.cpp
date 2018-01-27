#include "MainWindow.h"


bool Find_Sudoku_Board_Boundary(cv::Mat& rawMat, std::vector<cv::Point2f>& boundPts, cv::Mat& fgMat, cv::Mat& contourMat, uchar debug=0);
void Draw_Sudoku_Square_Centers(cv::Mat& contourMat, std::vector<cv::Point2f>& boundPts);
bool Get_Sudoku_Cell_Digit(cv::Mat& cellMat, cv::Mat& digitImg);
bool Fix_Sudoku_Corner_Order(std::vector<cv::Point2f>& unorderedPts, std::vector<cv::Point2f>& orderedPts);


MainWindow::MainWindow(QWidget* _parent)
{
    this->setParent(_parent);

    int debug = 1;
    writeMode = WRITE_MODE::APPEND;
    std::string filePath = "../data/sudoku_digits_01.png";
    std::string trainPath = "train.png";
    std::string responsePath = "response.png";

//    bool rotate90DegRight=false;
//    bool rotate90DegLeft=false;


    cv::Mat rawMat = cv::imread(filePath, CV_LOAD_IMAGE_GRAYSCALE);
    if (rawMat.empty())
        return;

    //resize input image into constant size. Mostly to make blur sizes consistent
    //this could be removed and blurring scaled... it's pretty slow to blur(80,80), and 4800x3600 px image. Faster to resize to 1200x900 + blur 20
    //cv::Size rawSize = cv::Size(rawMat.cols, rawMat.rows);
    int dim = std::sqrt(rawMat.cols*rawMat.cols + rawMat.rows*rawMat.rows);
    double scl = 1500.0/dim;  //want around x pixels for board diagonal? like 1200x900 ish
    cv::Mat sclMat;
    cv::resize(rawMat, sclMat, cv::Size(), scl, scl );
    //cv::GaussianBlur(sclMat, sclMat, cv::Size(3,3), 1, 1);  //maybe not helpful

    std::vector<cv::Point2f> fourCorners;
    cv::Mat contourMat, fgMat;
    if (!Find_Sudoku_Board_Boundary(sclMat, fourCorners, fgMat, contourMat, debug))
    {
        std::cout << "Oh noes! failure - could not find 4 corners of sudoku board :(" << std::endl;
        return;
    }

    //this is not necessary - maps sudoku space [0-9,0-9] into corresponding sudoku boundaries in input image and draws circle at center of each square
    if (debug) Draw_Sudoku_Square_Centers(contourMat, fourCorners);


    //test warp visually ### un-warps the input image so the sudoku board lies in a nice square mat
    //where the mat/board boundaries are one and the same (because input image will not be perfectly in-plane and cropepd to board)
    int cellDim = 20;
    int cellPadding = 2;
    int cellTotSize = cellDim+2*cellPadding;
    int transformSize = 9*(cellTotSize);  //252=28, 216=24, 576=64

    //this gets wonky if sudoku is significantly turned. in that case need to check that we warped it right-side-up
    //i think the way to do that is check if H*(fourCorners[0].x, fourCorners[0].y, 1) is ~(0,0)
    //if not maybe we just cycle the points in map points and try again?
    std::vector<cv::Point2f> mapPts = {cv::Point2f(0,0), cv::Point2f(transformSize,0), cv::Point2f(transformSize,transformSize), cv::Point2f(0,transformSize)};
    cv::Mat H = cv::findHomography(fourCorners, mapPts );

//    if (debug)
//    {
//        std::vector<cv::Point2f> transfCorners;
//        cv::perspectiveTransform(fourCorners,transfCorners,H);
//        std::cout << fourCorners << "  -->  " << transfCorners << std::endl;
//    }

//    double* H0 = H.ptr<double>(0), *H1 = H.ptr<double>(1);
//    if (abs(H0[0]*fourCorners[0].x + H0[1]*fourCorners[0].y + H0[2]) > cellDim || abs(H1[0]*fourCorners[0].x + H1[1]*fourCorners[0].y + H1[2]) > cellDim )


//    double rotData[9] = {0, -1, transformSize-2,  1, 0, -2,  0, 0, 1};
//    cv::Mat rotMat = cv::Mat(3,3, CV_64F, rotData);
//    double rotData2[9] = {0, 1, -2,  -1, 0, transformSize-2,  0, 0, 1};
//    cv::Mat rotMat2 = cv::Mat(3,3, CV_64F, rotData2);

    //bgMat?
    cv::Mat warpMat;
    cv::warpPerspective(fgMat, warpMat, H, cv::Size(transformSize,transformSize) ); // warpInput.size() );  //warpInput

//    if (rotate90DegRight)
//        cv::warpPerspective(warpMat, warpMat, rotMat, cv::Size(transformSize,transformSize) ); // warpInput.size() );  //warpInput
//    else if (rotate90DegLeft)
//        cv::warpPerspective(warpMat, warpMat, rotMat2, cv::Size(transformSize,transformSize) ); // warpInput.size() );  //warpInput


    if (debug)
    {
        std::cout << "H:\n" << H << std::endl << std::endl;
        //std::cout << "R:\n" << rotMat << std::endl << std::endl;
        //std::cout << "H-post:\n" << H << std::endl << std::endl;

        cv::namedWindow("warp",cv::WINDOW_NORMAL);
        cv::imshow("warp", warpMat);
        cv::waitKey(0);
    }

    //for training, we have trainMat: the doctored but raw images of each digit that another program can load for training (still requires processing, like reshaping and maybe feature extraction for training). then responseMat: the correct digit in each image (ready to use)
    cv::Mat trainMat, responseMat;
    if (writeMode == APPEND)
    {
        trainMat = cv::imread(trainPath, CV_LOAD_IMAGE_GRAYSCALE);
        responseMat = cv::imread(responsePath, CV_LOAD_IMAGE_GRAYSCALE);
    }

    //now extract digits from undistorted image of sudoku board
    for (size_t testRow=0; testRow<9; ++testRow)
    {
        for (size_t testCol=0; testCol<9; ++testCol)
        {
            cv::Rect cellRect(cellTotSize*testCol + cellPadding + 1, cellTotSize*testRow + cellPadding + 1, cellDim, cellDim);  //### if change cellDim change this +offset!

            //seems to be a bug in findContours for opencv 3.1. can't find contours for mat defined as an ROI of another map. So we need to make a hard copy
            cv::Mat cellMat, digitMat;
            warpMat(cellRect).copyTo(cellMat);

            if (!Get_Sudoku_Cell_Digit(cellMat, digitMat))
                continue;

            //cv::imshow("test detect image", cellMat);
            cv::imshow("combo", digitMat);

            //cv::imwrite("dig_img "+std::to_string(testRow)+"-"+std::to_string(testCol)+".png", digitMat);
            char c = cv::waitKey(0);

            if (c == 27)  //escape == 27
                return;

            if (c == 32)  //space == 32
                continue;

            uchar trueVal = c-48;  //assuming c is "0"-"9"

            trainMat.push_back(digitMat);
            responseMat.push_back(trueVal);
        }
    }

    cv::imwrite("train.png",trainMat);
    cv::imwrite("response.png",responseMat);
}



bool Find_Sudoku_Board_Boundary(cv::Mat& sclMat, std::vector<cv::Point2f>& boundPts, cv::Mat& fgMat, cv::Mat& contourMat, uchar debug)
{
    double blurSigma = sqrt(sclMat.cols*sclMat.cols + sclMat.rows*sclMat.rows) / 50;

    //correct for shading by bluring original image and subtract.
    cv::Mat bgMat;
    cv::GaussianBlur(sclMat,bgMat,cv::Size(),blurSigma,blurSigma); //cv::GaussianBlur(rawMat,bgMat,cv::Size(),55,55);

    //bgMat we're trying to detect dark lines/numbers in original image, so blur-original is >0 where original is dark
    fgMat = bgMat-sclMat;

//    cv::Mat fgMat = cv::Mat(bgMat.rows, bgMat.cols, CV_8U);
    //cv::Mat fgMat;
    cv::threshold(fgMat,fgMat,1,255, CV_8U);    //dark lines in original are 255 in fgMat, other is 0
//    cv::adaptiveThreshold() //I'm basically doing adaptive threshold but wanted to keep intermediary bgMat

    //identify the sudoku board via connected component analysis (finding the contours and looking at their props)
    //could also do like morphological close before findContours to try and fill gaps in border. Hasn;t been necessary yet but potential mode of failure
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(fgMat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (debug)
    {
        double minVal,maxVal;
        cv::minMaxIdx(bgMat, &minVal, &maxVal, nullptr, nullptr);

        //std::cout << "image size: " << rawSize << std::endl;
        //std::cout << "rescale factor: " << scl << std::endl;
        std::cout << "min: " << minVal << "    max: " << maxVal << std::endl;
        std::cout << "contours: " << contours.size() << std::endl;

        //draw our board detection boundary in color on top of input bw image
        cv::cvtColor(sclMat, contourMat, cv::COLOR_GRAY2BGR );

        cv::imshow("board pic", sclMat);
        cv::imshow("bg", bgMat);
        cv::imshow("fg", fgMat);
        //cv::waitKey(0);
    }

    //look for our sudoku. It should be roughly a square (possibly rotated and slightly warped). should also be fairly large, hopefull one of larger rects
    double bestFitVal = 0;
    size_t bestContIdx = 0;
    for (size_t contNo=0; contNo<contours.size(); ++contNo)
    {
        cv::Moments moms = cv::moments(contours[contNo]);
        if (moms.m00 < 500)       //board needs to be pretty big to read the numbers. weed out obvious non-boards
            continue;

        //ideally rotRect.area ~= m00 -> contour is a rectangle-ish
        cv::RotatedRect rotRect = cv::minAreaRect(contours[contNo]);
        cv::Rect brRect = rotRect.boundingRect();
        double rArea = rotRect.size.area();
        double areaDiff = 1 - (std::abs(moms.m00 - rArea) / rArea );
        double aspectDiff = 1 - (std::abs(brRect.width - brRect.height) / std::max(brRect.width, brRect.height));
        double fitVal = std::sqrt(rArea) * areaDiff * areaDiff * aspectDiff;  //some value I think is reasonable to determine the fitness of a contour as a sudoku box. certainly not machine-learning/AI level of assessment, I know

        if (fitVal > bestFitVal)
        {
            bestContIdx = contNo;
            bestFitVal = fitVal;
        }

        if (debug)
        {
            std::cout << "c#: " << contNo << "\t";
            //std::cout << "rect: " << bRect << "\t";
            std::cout << "area: " << int(moms.m00) << "  \t";
            std::cout << "rotA: " << int(rArea) << "\t";
            std::cout << "rotR: " << rotRect.angle << "<  " << rotRect.center << "\t";
            std::cout << "fit: " << int(fitVal) << std::endl;

            cv::drawContours(contourMat, contours, contNo, cv::Scalar(255,255,255),1, cv::LINE_8);
        }
    }

    //approximate full contour as a polygon. hopefully this will be a quadrilateral...
    //if not we should iterate with different epsilon until boundCnt==4, but not implementing yet (this works for all my test images)
    cv::RotatedRect bestRotRect = cv::minAreaRect(contours[bestContIdx]);
    double epsilon = std::sqrt(bestRotRect.boundingRect().area())/5;  // sqrt(area)/5 ~=1/20 perimeter? am i doing this right? On test image there's quite a large range that gives the expected 4pts, like 1/2th perim to 1/200th perim
    std::vector<cv::Point2f> tempBoundPts;
    cv::approxPolyDP(contours[bestContIdx], tempBoundPts, epsilon, true);
    size_t boundCnt = tempBoundPts.size();

    //check that boundPts is correctly oriented - we assume upper left bound point is upper left sudoku pt (note assumption), and require points be listed in counterclockwise order
    Fix_Sudoku_Corner_Order(tempBoundPts, boundPts);

    //draw contours and boundaries on copy of input image if debugging
    if (debug)
    {
        std::cout << "\n approx: " << boundCnt << std::endl;
        for (cv::Point2f& pt : boundPts)
            std::cout << pt << "\t";
        std::cout << std::endl;

        cv::Point2f rotRectPts[4];
        bestRotRect.points(rotRectPts);
        //draw best rotated rect approx of board. Ordered by ascending brightness
        for (size_t ptNo=0; ptNo<4; ++ptNo)
        {
            //std::cout << "pt: " << rotRectPts[ptNo] << std::endl;
            cv::line(contourMat, rotRectPts[ptNo], rotRectPts[(ptNo+1)%4], cv::Scalar(0,0,63+64*ptNo), 2);
        }

        //cv::Rect bestRect = cv::boundingRect(contours[bestContIdx]);
        //cv::rectangle(contourMat, bestRect, cv::Scalar(0,255,0), 2);
        cv::drawContours(contourMat, contours, bestContIdx, cv::Scalar(255,0,0), 2);

        //draw the boundary we'll actually use: polygon approximation of contour. Ordered by ascending brightness
        for (size_t ptNo=0; ptNo<boundCnt; ++ptNo)
        {
            //std::cout << "pt: " << rotRectPts[ptNo] << std::endl;
            cv::line(contourMat, boundPts[ptNo], boundPts[(ptNo+1)%boundCnt], cv::Scalar(0,63+64*ptNo,0), 2);
        }

        cv::namedWindow("contours",cv::WINDOW_NORMAL);
        cv::imshow("contours", contourMat);
        cv::waitKey(0);
    }

    return (boundCnt==4);
}




void Draw_Sudoku_Square_Centers(cv::Mat& contourMat, std::vector<cv::Point2f>& boundPts)
{
    std::vector<cv::Point2f> srcPts = {cv::Point2f(0,0), cv::Point2f(0,9), cv::Point2f(9,9), cv::Point2f(9,0)};
    cv::Mat Q = cv::findHomography(srcPts, boundPts);
    std::cout <<"\nBoard-to-image homography:\n" << Q << std::endl;

    //create vector with bounds of sudoku in sudoku space (0-9,0-9)
    //std::vector<std::vector<cv::Point2f>> inputCenterPts(9,std::vector<cv::Point2f>(9) );  //initialize vector of vectors example
    std::vector<cv::Point2f> inputCenterPts(81);
    for (size_t row=0; row<9; ++row)
    {
        for (size_t col=0; col<9; ++col)
        {
            //inputCenterPts[row][col] = cv::Point2f(col+.5, row+.5);
            inputCenterPts[9*row+col] = cv::Point2f(col+.5, row+.5);
        }
    }

    //transform sudoku space points into input-image/contourMat space. sudoku squares are centered at (.5+M , .5+N) in Q-transformed space
    //std::cout << "perspective tranform" << std::endl;
    std::vector<cv::Point2f> outputCenterPts(81);
    cv::perspectiveTransform(inputCenterPts, outputCenterPts,Q);
    for (size_t row=0; row<9; ++row)
    {
        for (size_t col=0; col<9; ++col)
        {
            //std::cout << outputCenterPts[9*row+col] << "\t";
            //std::cout << outputCenterPts[row][col] << "\t";
            cv::circle(contourMat, outputCenterPts[9*row+col], 3, cv::Scalar(0,255,255), -1);
        }
        //std::cout << std::endl;
    }

    cv::imshow("contours", contourMat);
    cv::waitKey(0);
}



bool Get_Sudoku_Cell_Digit(cv::Mat& cellMat, cv::Mat& digitImg)
{
    if (cv::mean(cellMat)[0] < 24)  //binarized to 0/255, so this says > 1/16 pixels are foreground/number
        return false;

    int cellW = cellMat.cols;
    int cellH = cellMat.rows;
    //this bit not needed if using fgImg as base for earp (fgImg is threshoded so 0 or 255, bgImg, is not normlized and would need this bit);
    //double minVal,maxVal,sclToMax;
    //cv::minMaxIdx(testDetectMat, &minVal, &maxVal, nullptr, nullptr);
    //sclToMax = 255/maxVal;
    //testDetectMat *= sclToMax;

    double maxContArea=0;
    int maxCont = -1;
    std::vector<std::vector<cv::Point>> cellContours;
    cv::findContours(cellMat, cellContours, cv::RetrievalModes::RETR_EXTERNAL, cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE); //actually don't need hierarchy
//    std::vector<cv::Vec4i> hierarchy;
//    cv::findContours(cellMat, cellContours, hierarchy, cv::RetrievalModes::RETR_LIST, cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE);

    for (size_t contNo=0; contNo<cellContours.size(); ++contNo)
    {
        cv::Moments cellMoms = cv::moments(cellContours[contNo]);    //use these as fit params/vars?
        if (cellMoms.m00 > (3*cellW*cellH/4))  //contour is pixels around edge and thus greater than or equal to size of entire mat...
            continue;

        if (cellMoms.m00 > maxContArea)
        {
            maxContArea = cellMoms.m00;
            maxCont = contNo;
            //cv::Point centPt(cellMoms.m01/cellMoms.m00, cellMoms.m10/cellMoms.m00);
        }
    }

    if (maxContArea < cellW*cellH/16)
        return false;

    cv::Rect bRect = cv::boundingRect(cellContours[maxCont]);
//    std::cout << bRect << std::endl;

//    size_t heirNo = maxCont;
    cv::Mat filtContMat = cv::Mat::zeros(cellW, cellH, CV_8UC1);
    cv::drawContours(filtContMat,cellContours,maxCont,cv::Scalar(255,255,255),cv::FILLED,cv::LINE_8);
//    cv::drawContours(redrawContMat,cellContours,maxCont,cv::Scalar(255,255,255),cv::FILLED,cv::LINE_8,hierarchy);

    //remove extraneous contours not part of digit (like grid lines bleeding into box around edges)
    cv::bitwise_and(cellMat, filtContMat, filtContMat);

    //digit generally isn't centered in cellMat, but now we have the contour we can shift it so bounding box around digit is centered
    digitImg = cv::Mat::zeros(cellW, cellH, CV_8UC1);
    cv::Rect centeredRect((cellW-bRect.width)/2, (cellH-bRect.height)/2, bRect.width, bRect.height);
    filtContMat(bRect).copyTo( digitImg(centeredRect) );

    //cv::Rect centeredRect((cellDim-bRect.width+1)/2, (cellDim-bRect.height+1)/2, bRect.width, bRect.height);
    //centeredMat(centeredRect) = filtContMat(bRect);
    return true;
}

//we want ordered pts to always follow same direction
//looking at cv::Mat with 0,0 in upper left, the orientation is clockwise
//looking at like cartesian coords with 0,0 in lower left, the orientation is counterclockwise
//this is not at all general or elegant :(
bool Fix_Sudoku_Corner_Order(std::vector<cv::Point2f>& unorderedPts, std::vector<cv::Point2f>& orderedPts)
{
    orderedPts.clear();
    orderedPts.reserve(4);

    //first point is upper left. actually it's kind of arbitrary assuming board is roughly facing natural direction in raw image
    cv::Point2f* upLeft;
    float minDistSqr = 999999.f;
    for (cv::Point2f& pt : unorderedPts)
    {
        float distSqr = pt.x*pt.x + pt.y*pt.y;
        if (distSqr < minDistSqr)
        {
            minDistSqr = distSqr;
            upLeft = &pt;
        }
    }

    orderedPts.push_back(*upLeft);

    //arbitrary point-1 (0,99999), point0 is upLeft, point1 is pt. Find pt with max angle
    double maxAng = 0;
    cv::Point2f* upRight;
    //### a needs to be flipped in direction: both vectors should originate at the last known point
    //cv::Vec2f a = cv::Vec2f( upLeft->x - 0,upLeft->y - 99999);  // -1->0 = [0]-[-1]
    cv::Vec2f a = cv::Vec2f( 0 - upLeft->x, 99999 - upLeft->y);  // 0->-1 = [-1]-[0]
    cv::Vec2f b;
    for (cv::Point2f& pt : unorderedPts)
    {
        if (&pt == upLeft)
            continue;

        b = cv::Vec2f(pt.x-upLeft->x, pt.y-upLeft->y); //0->1 = [1]-[0]
        double dot = a.dot(b);  //a.b?
        double mag = sqrt(a.dot(a)*b.dot(b));
        double ang = std::acos(dot/mag);
        if (ang>maxAng)
        {
            maxAng = ang;
            upRight = &pt;
        }
    }

    orderedPts.push_back(*upRight);

    maxAng = 0;
    cv::Point2f* downRight;
    a = cv::Vec2f( upLeft->x - upRight->x, upLeft->y - upRight->y);  // 1->0 = [0]-[1]
    for (cv::Point2f& pt : unorderedPts)
    {
        if (&pt == upLeft || &pt == upRight)
            continue;

        b = cv::Vec2f(pt.x-upLeft->x, pt.y-upLeft->y); //2->3
        double dot = a.dot(b);  //a.b?
        double mag = sqrt(a.dot(a)*b.dot(b)); // ||a||*||b||
        double ang = std::acos(dot/mag);  // arccos(a.b / ||a||*||b||) = theta?
        if (ang>maxAng)
        {
            maxAng = ang;
            downRight = &pt;
        }
    }

    orderedPts.push_back(*downRight);
    cv::Point2f* downLeft;

    for (cv::Point2f& pt : unorderedPts)
    {
        if (&pt == upLeft || &pt == upRight || &pt == downRight)
            continue;

        downLeft = &pt;
    }

    orderedPts.push_back(*downLeft);

    return true;
}
