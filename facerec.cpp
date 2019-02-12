#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

#include <iostream>
#include <fstream>
#ifdef __GNUC__
#include <experimental/filesystem> // Full support in C++17
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::tr2::sys;
#endif

int main(const int argc, char *argv[])
{
	// Declare cascade locations
	const std::string face_cascade_location = "../../opencv3.2/cvroot/etc/haarcascades/haarcascade_frontalface_default.xml";
	// Declare images and labels vectors
	std::vector<cv::Mat> images;
	std::vector<int>     labels;

	// Iterate through all subdirectories, looking for .pgm files
	const fs::path p(argc > 1 ? argv[1] : "../att_faces");
	for (const auto &entry : fs::recursive_directory_iterator{ p }) {
	
		if (entry.path().extension() == ".pgm") {
			std::string str = entry.path().parent_path().stem().string(); // s26 s27 etc.
			int label = atoi(str.c_str() + 1); // s1 -> 1
			images.push_back(cv::imread(entry.path().string().c_str(), 0));
			labels.push_back(label);
		}
	}

	// Assign the image width and height using the first image in the database - They are all uniform so this is acceptable.
	const int im_width = images[0].cols;
	const int im_height = images[0].rows;
	// Declare EigenFaceRecogniser for facial recognition
	cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::createEigenFaceRecognizer();
	// Train the facial recogniser against the database of faces.
	std::cout << "Training against faces in database...";
	model->train(images, labels);

	// Assign cascade classifier from file, if it fails an error message is returned.
	cv::CascadeClassifier face_cascade, eyes_cascade;
	if (!face_cascade.load(face_cascade_location)) { printf("--(!)Error loading face cascade\n"); return -1; };

	// Video capture declared, argument is the camera id, if they video fails to open then an error message is output.
	cv::VideoCapture vid_in(0);  
	if (!vid_in.isOpened()) {
		std::cout << "error: Camera 0 could not be opened for capture.\n";
		return -1;
	}
	// Matrix declared to store the frame from the camera.
	cv::Mat frame;
	// Begin infinite loop
	for (;;) {
		// Assign the current video in to the frame Matrix.
		vid_in >> frame;
		// Clone the current frame
		cv::Mat original = frame.clone();
		// Convert the current frame to grayscale
		cv::Mat gray;
		cvtColor(original, gray, cv::COLOR_BGR2GRAY);
		// Find the faces in the frame
		std::vector< cv::Rect > faces;
		// Find faces within the gray frame and deposit them into the faces vector
		face_cascade.detectMultiScale(gray, faces);

		// Inner loop to iterate over the faces in the vector
		for (const auto& i : faces)
		{
			// Process each face
			cv::Rect face_i = i;
			cv::Mat face = gray(face_i);

			// Resize down to the individual face
			cv::Mat face_resized;
			cv::resize(face, face_resized, cv::Size(im_width, im_height), 1.0, 1.0, cv::INTER_CUBIC);

			// Prediction calculation
			int predicted_label = -1;
			double predicted_confidence = 0.0;
			model->predict(face_resized, predicted_label, predicted_confidence);
			// Draw a blue rectangle around the detected face
			rectangle(original, face_i, cv::Scalar(0, 0, 255, 255), 5);
			// Create the text we will annotate the box with, displays the predicted confidence of the object being a face
			const int rounded_prediction = round(predicted_confidence);
			std::string box_text = cv::format("Prediction = %d", rounded_prediction);
			if (rounded_prediction > 6000)
			{
				box_text.insert(0, "Jack Smith - ");
			}
			// Calculate the position for annotated text max ensures against erroneous values
			const auto pos_x = std::max(face_i.tl().x - 10, 0);
			const auto pos_y = std::max(face_i.tl().y - 10, 0);
			// Put the text onto the image at the specified point
			putText(original, box_text, cv::Point(pos_x-100, pos_y), cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(0, 255, 255, 255), 2.5);
			cv::namedWindow("Cropped Image", cv::WINDOW_AUTOSIZE); // new window
			cv::imshow("Cropped Image", face_resized); // output the greyscaled cropped image in the new window
		}
		// Show the result:
		cv::imshow("Facial Recognition Test", original);
		const auto key = (char)cv::waitKey(20);
		// Exit this loop on escape press
		if (key == 27)
		{
			break;
		}
	}
	vid_in.release();
	return 0;
}
