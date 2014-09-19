#include "conversions.h"
#include <math.h>

sensor_msgs::Image conversions(cv::Mat mat)
{
    sensor_msgs::Image sensorImg;

    sensorImg.height = mat.rows;
    sensorImg.width = mat.cols;
    sensorImg.encoding = sensor_msgs::image_encodings::BGR8;
    sensorImg.is_bigendian = false;
    sensorImg.step = mat.step;
    size_t size = mat.step * mat.rows;
    sensorImg.data.resize(size);
    memcpy((char*)(&sensorImg.data[0]), mat.data, size);

    return sensorImg;
}

cv::Mat conversions(const sensor_msgs::ImageConstPtr& image)
{
    cv::Mat x;
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        if(image->encoding == "bgr8" ){
            cv_ptr = cv_bridge::toCvCopy(image,sensor_msgs::image_encodings::BGR8);
        }
        if(image->encoding == "32FC1"){
            cv_ptr = cv_bridge::toCvCopy(image,sensor_msgs::image_encodings::TYPE_32FC1);
        }
    }

    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception returning Null image: %s",  e.what());
        return x;
    }
    return cv_ptr->image;
}

double StdDeviation::CalculateMean()
{
    double sum = 0;
    for(int i = 0; i < max; i++)
    sum += value[i];
    return (sum / max);
}

double StdDeviation::CalculateVariane()
{
    mean = CalculateMean();

    double temp = 0;
    for(int i = 0; i < max; i++)
    {
        temp += (value[i] - mean) * (value[i] - mean) ;
    }
    return temp / max;
}

double StdDeviation::CalculateSampleVariane()
{
    mean = CalculateMean();

    double temp = 0;
    for(int i = 0; i < max; i++)
    {
        temp += (value[i] - mean) * (value[i] - mean) ;
    }
    return temp / (max - 1);
}

int StdDeviation::SetValues(double *p, int count)
{
    if(count > 1000)
    return -1;
    max = count;
    for(int i = 0; i < count; i++)
        value[i] = p[i];
    return 0;
}

double StdDeviation::GetStandardDeviation()
{
    return sqrt(CalculateVariane());
}

double StdDeviation::GetSampleStandardDeviation()
{
    return sqrt(CalculateSampleVariane());
}
