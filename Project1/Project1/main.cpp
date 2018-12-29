#include <opencv2/opencv.hpp>
#include <iostream>
#include <Windows.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

//모니터 캡처
Mat hwnd2mat(HWND hwnd)
{
	HDC hwindowDC, hwindowCompatibleDC;

	int height, width, srcheight, srcwidth;
	HBITMAP hbwindow;
	Mat src;
	BITMAPINFOHEADER  bi;

	hwindowDC = GetDC(hwnd);
	hwindowCompatibleDC = CreateCompatibleDC(hwindowDC);
	SetStretchBltMode(hwindowCompatibleDC, 3);

	RECT windowsize;    // get the height and width of the screen
	GetClientRect(hwnd, &windowsize);

	srcheight = windowsize.bottom / 0.78;
	srcwidth = windowsize.right / 0.9;
	height = windowsize.bottom / 1;  //change this to whatever size you want to resize to
	width = windowsize.right / 1;

	src.create(height, width, CV_8UC4);

	// create a bitmap
	hbwindow = CreateCompatibleBitmap(hwindowDC, width, height);
	bi.biSize = sizeof(BITMAPINFOHEADER);    //http://msdn.microsoft.com/en-us/library/windows/window/dd183402%28v=vs.85%29.aspx
	bi.biWidth = width;
	bi.biHeight = -height;  //this is the line that makes it draw upside down or not
	bi.biPlanes = 1;
	bi.biBitCount = 32;
	bi.biCompression = 0;
	bi.biSizeImage = 0;
	bi.biXPelsPerMeter = 0;
	bi.biYPelsPerMeter = 0;
	bi.biClrUsed = 0;
	bi.biClrImportant = 0;

	// use the previously created device context with the bitmap
	SelectObject(hwindowCompatibleDC, hbwindow);
	// copy from the window device context to the bitmap device context
	StretchBlt(hwindowCompatibleDC, 0, 0, width, height, hwindowDC, 0, 0, srcwidth, srcheight, 0x00CC0020); //change SRCCOPY to NOTSRCCOPY for wacky colors !
	GetDIBits(hwindowCompatibleDC, hbwindow, 0, height, src.data, (BITMAPINFO *)&bi, 0);  //copy from hwindowCompatibleDC to hbwindow

	// avoid memory leak
	DeleteObject(hbwindow);
	DeleteDC(hwindowCompatibleDC);
	ReleaseDC(hwnd, hwindowDC);

	return src;
}

int main(int argc, char **argv)
{
	HWND hwndDesktop = GetDesktopWindow();
	int key = 0;

	//esc 누르기 전까지 모니터를 캡처한다
	//while (key != 27)
	{
		//tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
		//ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
		//ocr->SetPageSegMode(tesseract::PSM_AUTO);
		Mat src = hwnd2mat(hwndDesktop);
		Mat foodCount(src, Rect(330, 750, 160, 100));
		foodCount = imread("C:\\Users\\Quote\\Desktop\\img_gray.jpg", 0);
		//ocr->SetImage(foodCount.data, foodCount.cols, foodCount.rows, 3, foodCount.step);
		//cout << string(ocr->GetUTF8Text()) << endl;
		//imshow("output", foodCount);
		//imwrite("C:\\Users\\Quote\\source\\repos\\Project1\\img_gray.jpg", foodCount);
		key = waitKey(0);
	}
	
}
