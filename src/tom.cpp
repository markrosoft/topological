#include <stdio.h>
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <set>
#include <stdio.h>
#include <math.h>

#include <ros/ros.h>
#include <ros/time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include "Common.h"
#include "FindCameraMatrices.h"
#include "Triangulation.h"
#include "conversions.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace std;
using namespace cv;

class KinectSensor {

    public:

        KinectSensor(ros::NodeHandle* _n)
        {
            this->n = *_n;
            this->sub = this->n.subscribe("/head_xtion/rgb/image_color", 1, &KinectSensor::callBack, this);
            cloudPub = n.advertise<sensor_msgs::PointCloud2> ("/marks_feature_cloud", 1);
            cloudPub1 = n.advertise<sensor_msgs::PointCloud2> ("/marks_feature_cloud1", 1);
            average.push_back(0); average.push_back(0); average.push_back(0);
            count=0;
        }

        void callBack(const sensor_msgs::ImageConstPtr& msg)
        {
            cv::Mat img_1 = cv::imread( "/home/mark/frame0000.jpg", CV_LOAD_IMAGE_GRAYSCALE );  //DriveBy1
            cv::Mat img_2 = conversions(msg);

            if( !img_1.data || !img_2.data )
            { return ; }

            //-- Step 1: Detect the keypoints using SURF Detector
            int minHessian = 400;
            cv::SurfFeatureDetector detector( minHessian );
            std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
            detector.detect( img_1, keypoints_1 );
            detector.detect( img_2, keypoints_2 );

            //-- Step 2: Calculate descriptors (feature vectors)
            cv::SurfDescriptorExtractor extractor;
            cv::Mat descriptors_1, descriptors_2;
            extractor.compute( img_1, keypoints_1, descriptors_1 );
            extractor.compute( img_2, keypoints_2, descriptors_2 );

            std::vector<std::vector <cv::DMatch> > preProcessedMatches;


            cv::BFMatcher matcher(cv::NORM_L2,  false);;

            matcher.knnMatch( descriptors_1, descriptors_2, preProcessedMatches, 2);

            std::vector< cv::DMatch > matches;

            float DISTANCE_FACTOR =  0.9;

            for(   int i=0; i < preProcessedMatches.size(); i++)
               {
                   if (preProcessedMatches[i].size() == 2)
                   {
                       if (preProcessedMatches[i][0].distance < preProcessedMatches[i][1].distance * 0.85)
                       {
                           DMatch match = DMatch(preProcessedMatches[i][0].queryIdx, preProcessedMatches[i][0].trainIdx, preProcessedMatches[i][0].distance);
                           //std::cout << preProcessedMatches[i][0].queryIdx <<"\t"<<preProcessedMatches[i][0].trainIdx << std::endl;

                           float difX = keypoints_1[preProcessedMatches[i][0].queryIdx].pt.x - keypoints_2[preProcessedMatches[i][0].trainIdx].pt.x;
                           float difY = keypoints_1[preProcessedMatches[i][0].queryIdx].pt.y - keypoints_2[preProcessedMatches[i][0].trainIdx].pt.y;
                           float dis = sqrt((difX*difX)+(difY*difY));

                           if(difY < 50 && difY > -50)    // .. this will disregard diagonal matches
                               matches.push_back(match);
                       }else{
                           DMatch match = DMatch(preProcessedMatches[i][0].queryIdx, -1, preProcessedMatches[i][0].distance);
                           //std::cout << preProcessedMatches[i][0].queryIdx <<"\t"<<preProcessedMatches[i][0].trainIdx << " <<" << std::endl;
                          // matches.push_back(match);
                       }
                   }
                   else if (preProcessedMatches[i].size() == 1)
                   {
                       DMatch match = DMatch(preProcessedMatches[i][0].queryIdx, preProcessedMatches[i][0].trainIdx, preProcessedMatches[i][0].distance);
                       matches.push_back(match);
                   }else{
                       DMatch match = DMatch(preProcessedMatches[i][0].queryIdx, -1, preProcessedMatches[i][0].distance);
                       matches.push_back(match);
                   }
               }

            std::cout << matches.size() << std::endl;


            //-- Draw matches
            cv::Mat img_matches;
            cv::drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches );
            //-- Show detected matches
            cv::namedWindow( "Matches", CV_WINDOW_NORMAL );
            cv::imshow("Matches", img_matches );
            cv::waitKey(30);


            //-- Step 4: calculate Fundamental Matrix
            std::vector<cv::Point2f>imgpts1,imgpts2;
            for( unsigned int i = 0; i<matches.size(); i++ )
            {

                imgpts1.push_back(keypoints_1[matches[i].queryIdx].pt);
                // trainIdx is the "right" image
                imgpts2.push_back(keypoints_2[matches[i].trainIdx].pt);
            }

            //double data[] = {525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0};//Camera Matrix
            double data[] = {526.60717328, 0.00000000, 318.52510740, 0.00000000, 526.60717328, 241.18145973, 0.00000000, 0.00000000, 1.00000000};
            cv::Mat K(3, 3, CV_64F, data);

            /*
            cv::Mat F =  cv::findFundamentalMat(imgpts1, imgpts2);//, cv::FM_RANSAC, 0.1, 0.99);
            cv::Mat_<double> E = K.t() * F * K; //acco  rding to HZ (9.12)
            */

            Mat Kinv(3, 3, CV_64FC1);
            Kinv = K.inv();

            Matx34d P(1, 0, 0, 0,
                      0, 1, 0, 0,
                      0, 0, 1, 0);
            Matx34d P1(1, 0, 0, 50,
                      0, 1, 0, 0,
                      0, 0, 1, 0);

            double dist[] = { 0.10265184, -0.18646532, 0.00000000, 0.00000000, 0.00000000};
            cv::Mat D(1, 5, CV_64F, dist);


            std::vector<cv::KeyPoint> pts1,pts2;  // convert using common.h
            KeyPointsToPoints(pts1, imgpts1);
            KeyPointsToPoints(pts2, imgpts2);

//
            std::vector<cv::KeyPoint> p1,p2;
            std::vector<CloudPoint> cloud, cloud1;

            bool x = FindCameraMatrices(K,Kinv,D,keypoints_1,keypoints_2, p1,p2,P,P1,matches,cloud,cloud1);
//K, Kinv, distcoeff, P, P1, pcloud, corresp);

            vector<CloudPoint> pcloud,pcloud1; vector<KeyPoint> corresp;
            TriangulatePoints(p1,p2, K, Kinv,D, P, P1, cloud, corresp);
            //  http://ksimek.github.io/2012/08/22/extrinsic/   << explains transformation vector

            if(x)
            {
            //average[0] += P1(0,3);
            //average[1] += P1(1,3);
            //average[2] += P1(2,3);

            average[0] = P1(0,3);
            average[1] = P1(1,3);
            average[2] = P1(2,3);

            count++;
            //std::cout << count << "\tcoooord XYZ: \t" << float(average[0] / count) << "\t" << average[1] / count << "\t" << average[2] / count << std::endl;
            std::cout << count << "\treal XYZ: \t" << average[0]  << "\t" << average[1]  << "\t" << average[2] << std::endl;
            }

            std::cout << cloud.size() << " " << cloud1.size() << std::endl;

            printf("%.54f\t%.54f\n",cloud[0].pt.x, cloud[0].pt.x);

            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster1(new pcl::PointCloud<pcl::PointXYZ>);

            for(int i = 0; i < cloud.size(); i ++)
            {
                pcl::PointXYZ temp,temp1;
                temp.x = cloud[i].pt.x;
                temp.y = cloud[i].pt.y;
                temp.z = cloud[i].pt.z;

                temp1.x = cloud1[i].pt.x+10;
                temp1.y = cloud1[i].pt.y;
                temp1.z = cloud1[i].pt.z;

                cloud_cluster->push_back(temp);
                cloud_cluster1->push_back(temp1);
            }

            ros::Time time_st = ros::Time::now();
            cloud_cluster->header.stamp = time_st.toNSec()/1e3;
            cloud_cluster->header.frame_id  = msg->header.frame_id ;

            cloud_cluster1->header.stamp = time_st.toNSec()/1e3;
            cloud_cluster1->header.frame_id  = msg->header.frame_id ;

            //cloud_cluster

            sensor_msgs::PointCloud2 cloud_out, cloud_out1;
            pcl::toROSMsg(*cloud_cluster, cloud_out);
            pcl::toROSMsg(*cloud_cluster1, cloud_out1);

            cloudPub.publish(cloud_out);
            cloudPub1.publish(cloud_out1);
            //ROS_ERROR("PUBLISHED!!");

        }
    protected:
        float count;
        std::vector<double > average;

        ros::Subscriber sub;
        ros::NodeHandle n;

        ros::Publisher  cloudPub, cloudPub1;
};


int main(int argc, char **argv)
{
    string arrStr[4] = {"/home/mark/Aisle1.txt", "/home/mark/Kitchen.txt", "/home/mark/WayPoint3.txt", "/home/mark/Aisle2.txt"};
    vector <vector <vector <float> > > data;


    for (string* p = &arrStr[0]; p != &arrStr[4]; ++p) {
        const char * c = p->c_str();

        vector <vector <float> > fileData;
        ifstream infile( c );

        while (infile)          // read in files
        {
            string s;
            if (!getline( infile, s )) break;

            istringstream ss( s );
            vector <float> record;

            while (ss)
            {
              string s;
              if (!getline( ss, s, ',' )) break;
              record.push_back(atof(s.c_str()));
            }

            fileData.push_back( record );
        }
        data.push_back(fileData);
        if (infile.eof())
            cout << *p <<  " <- read." <<  endl;
    }

    for(int f = 0; f < 4; f ++)             // bubble sort them
    {
        for(int i = 0; i < 49; i ++)
        {
            for(int j = 0; j < 49-i; j++)
            {
                float xtemp;
                if(data[f][j][0] > data[f][j+1][0]){
                    xtemp = data[f][j+1][0];
                    data[f][j+1][0] = data[f][j][0];
                    data[f][j][0] = xtemp;
                }

                float ytemp;
                if(data[f][j][1] > data[f][j+1][1]){
                    ytemp = data[f][j+1][1];
                    data[f][j+1][1] = data[f][j][1];
                    data[f][j][1] = ytemp;
                }

                float ztemp;
                if(data[f][j][2] > data[f][j+1][2]){
                    ztemp = data[f][j+1][2];
                    data[f][j+1][2] = data[f][j][2];
                    data[f][j][2] = ztemp;
                }
            }
        }
    }


    double temp[] = {data[0][25][2], data[0][25][1],         // GET MEDIAN
                     data[1][25][2], data[1][25][1],
                     data[2][25][2], data[2][25][1],
                     data[3][25][2], data[3][25][1]};
   
    cv::Mat v(4, 2, CV_64F, temp);

    double wpc[] = {0,0,0,-1,-1,-1,-1,0};
    cv::Mat b(4, 2, CV_64F, wpc);

    cv::Mat vt(2, 4, CV_64F);
    cv::transpose(v, vt);

    Mat c = Mat(b* vt).diag();

    Mat a0(1, 1, CV_64F);
    Mat a1(1, 1, CV_64F);
    Mat a2(1, 1, CV_64F);
    Mat b0(1, 1, CV_64F);
    Mat b1(1, 1, CV_64F);

    a0.at<double>(0) = 0;
    a1.at<double>(0) = 0;
    a2.at<double>(0) = 0;
    b0.at<double>(0) = 0;
    b1.at<double>(0) = 0;

    for(int i = 0; i < 4; i++)
    {
        Mat v0(1, 1, CV_64F, v.at<double>(i,0));
        Mat v1(1, 1, CV_64F, v.at<double>(i,1));
        Mat cEl(1, 1, CV_64F, c.at<double>(i));

        a0.at<double>(0) += ((cv::Mat)(v0.mul(v0))).at<double>(0);
        a1.at<double>(0) += ((cv::Mat)(v0.mul(v1))).at<double>(0);
        a2.at<double>(0) += ((cv::Mat)(v1.mul(v1))).at<double>(0);
        b0.at<double>(0) += ((cv::Mat)(v0.mul(cEl))).at<double>(0);
        b1.at<double>(0) += ((cv::Mat)(v1.mul(cEl))).at<double>(0);
    }
    Mat d = a0*a2-a1*a1;
d.inv();
    Mat x=(b0*a2-b1*a1)/d;
    Mat y=(-(b0)*a1+b1*a0)/d;
    std::cout << x << y<< std::endl;
     //cv::multiply(v.at<float>(0,0),v.at<float>(0,0));
     



    /*
  

a0=median(w0);
a1=median(w1);
a2=median(w2);
a3=median(w3);


v=[a0(:,3),-a0(:,1);
   a1(:,3),-a1(:,1);
   a2(:,3),-a2(:,1);
   a3(:,3),-a3(:,1)];

%b has coordinates of the waypoints
*/



/////////////// TOM'S CODES
/*
int numWaypoints = 3;

float vx[numWaypoints]; 
float vy[numWaypoints];
float bx[numWaypoints]; 
float by[numWaypoints];

for (int i=0;i<numWaypoints;i++){
    bx[i] = 0; //Waypoint i coord X
    by[i] = 1; //Waypoint i coord Y
    vx[i] = +temp[i][1];
    vy[i] = -temp[i][0];
}

float c[numWaypoints];
for (int i=0;i<numWaypoints;i++) c[i]=bx[i]*vx[i]+by[i]*by[i];

float a0,a1,a2,b0,b1;
a0=a1=a2=b0=b1=0;
for (int i=0;i<numWaypoints;i++){
a0+= vx[i]*vx[i];
a1+= vx[i]*vy[i];
a2+= vy[i]*vy[i];
b0+= vx[i]*c[i];
b1+= vy[i]*c[i];
}

Mat d=a0*a2-a1*a1;

Mat x =(+b0*a2-b1*a1)/d;
Mat y =(-b0*a1+b1*a0)/d;

std::cout <<  x << "\t" << y << std::endl;
*/
    std::exit(1);

    std::cout << "Running" << std::endl;

    ros::init(argc, argv, "test");
    ros::NodeHandle n;

    KinectSensor e(&n);
    ros::spin();

    return 0;
}
