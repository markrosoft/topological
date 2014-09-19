#ifndef CONVERSIONS_H
#define CONVERSIONS_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>

sensor_msgs::Image conversions(cv::Mat mat);
cv::Mat conversions(const sensor_msgs::ImageConstPtr&);

class StdDeviation
{

private:
    int max;
    double value[1000];
    double mean;

public:
    double CalculateMean();
    double CalculateVariane();
    double CalculateSampleVariane();
    int SetValues(double *, int);
    double GetStandardDeviation();
    double GetSampleStandardDeviation();
};


#endif

