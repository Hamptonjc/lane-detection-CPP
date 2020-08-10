#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

cv::Mat loadImg(std::string img_path){
	return cv::imread(img_path, cv::IMREAD_COLOR);
};

cv::Mat colorFilter(cv::Mat src){
	cv::Mat hls, yellowmask, whitemask, mask, masked;
	cv::cvtColor(src, hls, cv::COLOR_RGB2HLS);
	cv::inRange(hls, cv::Scalar(10,0,90), cv::Scalar(50,255,255), yellowmask);
	cv::inRange(hls, cv::Scalar(0,190,0), cv::Scalar(255,255,255), whitemask);
	cv::bitwise_or(yellowmask, whitemask, mask);
	cv::bitwise_and(src, src, masked, mask=mask);
	return masked;
};

cv::Mat ROI(cv::Mat src){
	int x = src.cols;
	int y = src.rows;
	cv::Point polygon_vertices[1][4];
	polygon_vertices[0][0] = cv::Point(0,y);
	polygon_vertices[0][1] = cv::Point(x,y);
	polygon_vertices[0][2] = cv::Point((int)std::round(0.55*x),(int)std::round(0.6*y));
	polygon_vertices[0][3] = cv::Point((int)std::round(0.45*x),(int)std::round(0.6*y));
	const cv::Point* polygons[1] = { polygon_vertices[0] };
	int n_vertices[] = { 4 };
	cv::Mat mask(y, x, CV_8UC1, cv::Scalar(0));
	int lineType = cv::LINE_8;
	cv::fillPoly(mask,polygons,n_vertices,1,cv::Scalar(255,255,255),lineType);
	cv::Mat masked_image;
	cv::bitwise_and(src, src, masked_image, mask=mask);
	return masked_image;
};

cv::Mat grayscale(cv::Mat img){
	cv::Mat gray_img;
	cv::cvtColor(img, gray_img, cv::COLOR_RGB2GRAY);
	return gray_img;
};

cv::Mat canny(cv::Mat img){
	cv::Mat edges;
	cv::Canny(grayscale(img), edges, 50, 120);
	return edges;
};


float vectorAverage(std::vector<float> input_vec){
	float average = std::accumulate(input_vec.begin(), input_vec.end(), 0.0)/input_vec.size();
	return average;
};

void drawLines(cv::Mat img, std::vector<cv::Vec4f> lines, int thickness = 5){
	cv::Scalar right_color = cv::Scalar(0,255,0);
	cv::Scalar left_color = cv::Scalar(0,0,255);
	std::vector<float> rightSlope, leftSlope, rightIntercept, leftIntercept;
	for (cv::Vec4f line : lines){
		float x1 = line[0];
		float y1 = line[1];
		float x2 = line[2];
		float y2 = line[3];
		float slope = (y1-y2)/(x1-x2);
		if (slope > 0.3){
			if (x1 > 500){
				float yintercept = y2 - (slope*x2);
				rightSlope.push_back(slope);
				rightIntercept.push_back(yintercept);
			};
		}
		else if (slope < -0.3){
			if (x1 < 600){
				float yintercept = y2 - (slope*x2);
				leftSlope.push_back(slope);
				leftIntercept.push_back(yintercept);
			};
		};
	};

	float left_intercept_avg = vectorAverage(leftIntercept);
	float right_intercept_avg = vectorAverage(rightIntercept);
	float left_slope_avg = vectorAverage(leftSlope);
	float right_slope_avg = vectorAverage(rightSlope);

	// for video write functionality to average slopes and intercepts over 30 frames

	//try{
	int left_line_x1 = (int)std::round((0.65*img.rows - left_intercept_avg)/left_slope_avg);
	int left_line_x2 = (int)std::round((img.rows - left_intercept_avg)/left_slope_avg);
	int right_line_x1 = (int)std::round((0.65*img.rows - right_intercept_avg)/right_slope_avg);
	int right_line_x2 = (int)std::round((img.rows - right_intercept_avg)/right_slope_avg);
	cv::Point line_vertices[1][4];
	line_vertices[0][0] = cv::Point(left_line_x1,(int)std::round(0.65*img.rows));
	line_vertices[0][1] = cv::Point(left_line_x2,img.rows);
	line_vertices[0][2] = cv::Point(right_line_x2,img.rows);
	line_vertices[0][3] = cv::Point(right_line_x1,(int)std::round(0.65*img.rows));
	const cv::Point* inner_shape[1] = { line_vertices[0] };
	int n_vertices[] = { 4 };
	int lineType = cv::LINE_8;
	cv::fillPoly(img, inner_shape,n_vertices,1,cv::Scalar(255,0,0),lineType);
	cv::line(img, cv::Point(left_line_x1, (int)std::round(0.65*img.rows)), cv::Point(left_line_x2, img.rows), left_color, 10);
	cv::line(img, cv::Point(right_line_x1, (int)std::round(0.65*img.rows)), cv::Point(right_line_x2, img.rows), right_color, 10);
	//catch(){
		//std::cout<<"function drawlines try block not executed"<<std::endl;

};

cv::Mat hough_lines(cv::Mat img, double rho, double theta, int threshold, double min_line_len,double max_line_gap){
	std::vector<cv::Vec4f> lines;
	cv::Mat line_img(img.rows, img.cols, CV_8UC3, cv::Scalar(0,0,0));
	cv::HoughLinesP(img, lines, rho, theta, threshold, min_line_len, max_line_gap);
	drawLines(line_img, lines);
	return line_img;
};

cv::Mat lineDetect(cv::Mat img){
	return hough_lines(img, 1, CV_PI/180, 10, 20, 100);
};

cv::Mat weighted_img(cv::Mat img, cv::Mat initial_img, double alpha = 0.8, double beta=1.0, double gamma = 0.0){
	cv::Mat weighted_img;
	cv::addWeighted(img, alpha, initial_img, beta, gamma, weighted_img);
	return weighted_img;
};


cv::Mat LaneDetection(cv::Mat src){
	cv::Mat color_masked, roi_img, canny_img, hough_img, final_img;
	color_masked = colorFilter(src);
	roi_img = ROI(color_masked);
	canny_img = canny(roi_img);
	hough_img = lineDetect(canny_img);
	final_img = weighted_img(hough_img, src);
	return final_img;
};

int main(){
	std::string video_path;

	std::cout<<"Please enter path to test video: "<<std::endl;
	std::cin>>video_path;

	cv::VideoCapture input_video(video_path);
	if(!input_video.isOpened()){
	    std::cout << "Error opening video stream or file" << std::endl;
	    return -1;
	}

	while(1){
		cv::Mat frame, processed_frame;

		input_video >> frame;

		if (frame.empty())
			break;

		processed_frame = LaneDetection(frame);
		cv::imshow("Output Video", processed_frame);

		char c=(char)cv::waitKey(25);
		if(c==27)
			break;
	}

	input_video.release();
	cv::destroyAllWindows();
	return 0;
};

