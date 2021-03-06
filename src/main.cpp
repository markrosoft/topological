#include <iostream>
#include <pcl/console/parse.h>
#include <ros/ros.h>
#include "std_srvs/Empty.h"
#include "topological_recovery/terminalService.h"

#include <ros/ros.h>
#include <ros/time.h>
#include <sensor_msgs/Image.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Pose.h>
#include <boost/foreach.hpp>
#include "conversions.h"
#include <fstream>
#include <dirent.h>
#include <algorithm>
#include <unistd.h>
#include <ctime>
#include <actionlib_msgs/GoalStatusArray.h>
#include <cstdlib>
#include <sstream>
#include <string>

#include "Common.h"
#include "FindCameraMatrices.h"
#include "Triangulation.h"

using namespace cv;
using namespace std;

#define NUM_WAYPOINTS 4
#define NUM_FUNDS  50

int numMatches  = NUM_WAYPOINTS;
int bestMatches[NUM_WAYPOINTS];
int bestIndex[NUM_WAYPOINTS];
double vx[NUM_WAYPOINTS][50];
double vy[NUM_WAYPOINTS][50];
int counts[NUM_WAYPOINTS];

int switchNum = 0;
double robotPose[3];

int myFundCount = -1;
bool fundChecker = true;

std::string headerFormating = "groundTruth\testimation\tangle\tmovement\tmovement\tmatches\t";

#define PI 3.14159265
#define DISTANCE_FACTOR 0.8

float PTU_UPPERLIMIT = 3;
float PTU_LOWERLIMIT = -3;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::JointState> MySyncPolicy;

struct feature_struct {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    double robotPose[3];
};

struct estimation
{
    std::string name;
    float angle;
    int matches;
};

char runOutputFile[] = "Estimation.txt";

class ptu_features
{
    public:
        std::string locationID;
        std::string groundTruth;
        std::ofstream outfile;
        std::vector<estimation> whereAmI;

        ptu_features(ros::NodeHandle* _n)                                                                   // Constructor
        {
            n = *_n;
            this->lost_sub = this->n.subscribe("/speak/status", 1, &ptu_features::lost_callBack, this);    // listen if robot is lost
            this->service = this->n.advertiseService("topological_string", &ptu_features::string_callBack, this);
            is_linda_lost = false;
            ptu.name.resize(2);
            ptu.position.resize(2);
            ptu.velocity.resize(2);

            ptu_pub = this->n.advertise<sensor_msgs::JointState>("/ptu/cmd", 1);
            isLost = this->n.advertise<std_msgs::String>("/topological/lostState", 1);

            myfile.open("lastGraph.txt");
            detector.hessianThreshold = 1000;
            ptu_start_angle = 1000; // init to number greater than 0-360
            f = 590;                // claims 525
            ptu_features::init();
            this->pose_sub = this->n.subscribe("/robot_pose", 10, &ptu_features::poseCallback, this);
            realRobot_pose = 1000;

            int cluster = 100;
            bow = new cv::BOWKMeansTrainer(cluster,cv::TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, FLT_EPSILON), 1, cv::KMEANS_PP_CENTERS );
            ForwardCamFrame = cv::Mat::zeros(1, 1, CV_32F);
            dirName = "~/";
        }

        void lost_callBack(const actionlib_msgs::GoalStatusArray msg)
        {
            if(msg.status_list.size() > 0)
            {
                std::string temp = msg.status_list[0].text;
                if(!temp.empty())
                {
                    if(is_linda_lost == false)
                    {
                        is_linda_lost = true;
                        //switchNum = 2;
                        std::cout << "Let the spin begin!" << std::endl;
                    }
                }
            }
        }

        bool string_callBack(topological_recovery::terminalService::Request &req, topological_recovery::terminalService::Response &res)
        {
            std::cout << req.file << "\t" << req.state << std::endl;
            //if(!req.state.empty())
            {
                std::string temp = req.state;
                if(temp == "reset")
                {
                    std::cout << "Reseting to defaults" << std::endl;
                    switchNum = 0;
                }
                if(temp == "record")
                {
                    std::cout << "Reseting to defaults" << std::endl;
                    switchNum = 1;
                }
                if(temp == "localise")
                {
                    switchNum = 2;  // 1 for init pictures
                }
                if(temp == "exit" || temp == "kill")
                {
                    std::exit(1);
                }
            }
            if(!req.file.empty())
            {
                groundTruth = req.file;
                std::cout << "New Goal State: " << groundTruth << std::endl;
                if(switchNum==1)
                    locationID=groundTruth;
            }

            res.response = "Done";
            return true;
        }

        void init()
        {
            average.clear();
            average.push_back(0); average.push_back(0); average.push_back(0);
            count = 0;
            subcount = 0;
            pointer = 0;
            estimatedAngle = 0;
            lastPtuAng = 1000;
            ptu_iterator = 45;
            ptu_pose = -180;
            ptu_stationary = 0.01; // Make initial
            ptuGoal = 1000;
            feature_sphere.keypoints.clear();
            feature_sphere.descriptors.release();
            is_linda_lost = false;
            bestMatch = 0;
        }

        void callback(const sensor_msgs::ImageConstPtr &img,  const sensor_msgs::JointStateConstPtr &ptu_state)
        {
            float PTUdif;
            switch (switchNum)
            {
            case 0:
            {
                init();
                break;
            }
            case 1:
            {
                if(this->read_features.size() == 0)
                {
                    std::stringstream ss;
                    ss<<this->groundTruth<< ".jpg";

                    cv::imwrite(ss.str(), conversions(img));
                }
                switchNum = 2;
                break;
            }
            case 2:
                if(ptuGoal == 1000)
                    moveCam(ptu_pose);
                PTUdif = 0;
                for (int i = 0;i<ptu_state->name.size();i++)
                {
                    if (ptu_state->name[i] == "pan") {
                        ptuAng = ptu_state->position[i] * 60;

                        PTUdif = ptu_state->position[i] > ptuGoal ? ptu_state->position[i] - ptuGoal : ptuGoal - ptu_state->position[i];
                    }
                }
                if(PTUdif < 0.001 || ptuGoal == 1000)
		{
			if(((double) (clock() - stationary_clock) / CLOCKS_PER_SEC) > ptu_stationary)
			{

				if(ptu_pose <= 180)
				{
					if(ptu_start_angle == 1000)
					{
						ptu_start_angle = ptuAng;
					}else{
						extractFeatures(conversions(img), feature_sphere, ptuAng);
					}
					stationary_clock = clock();
					moveCam(ptu_pose);
					if(ptu_pose==45)
						ForwardCamFrame = conversions(img);

					ptu_pose += ptu_iterator;
				}
				else    // ptu spin complete
				{
					std::cout << "the sweep is complete" << std::endl;
					if(this->read_features.size() > 0)
					{

						for(int i = 0; i < whereAmI.size(); i ++)
						{
							this->pointer = i;
							myfile << whereAmI[i].name << "\t";
							match(read_features[i], feature_sphere);
							myfile << "\n";
						}

						//best matches	TOM
						myFundCount = numMatches;
						for (int j = 0;j<numMatches;j++){	
							bestMatches[j] = -1;
							for(int i = 0; i < whereAmI.size(); i ++)
							{
								if(bestMatches[j] < whereAmI[i].matches)
								{
									bestMatches[j]=whereAmI[i].matches;
									bestIndex[j] = i;
								}
							}
							whereAmI[bestIndex[j]].matches = -1;
						}
						for (int j = 0;j<numMatches;j++) whereAmI[bestIndex[j]].matches = bestMatches[j];
						for (int j = 0;j<numMatches;j++) printf("MATCHES: %i %i\n",bestIndex[j],bestMatches[j]);
						switchNum = 5;

						/*                                locationID = whereAmI[bestMatch].name;
										  estimatedAngle = whereAmI[bestMatch].angle;

										  if(is_linda_lost || whereAmI[bestIndex[0]].matches < 1001)      // (100) Has linda said she is lost or returned bad matches?
										  {
										  is_linda_lost = true;
										  std::cout << "Not enough information to go on attempting fundamental matrix!" << std::endl;
										  if(fundChecker)
										  estimatedAngle = whereAmI[myFundCount].angle;

										  moveCam(((-int(estimatedAngle)-180) % 360)+180);

										  switchNum = 3;
										  }
										  else
										  {
										  outfile.open(runOutputFile, std::ios_base::app);
										  outfile << groundTruth << "\t" << locationID << "\t" <<  whereAmI[bestMatch].angle << "\t" << whereAmI[bestMatch].matches << "\tNA\tNA\t";
										  for(int i = 0; i < whereAmI.size(); i ++)
										  {
										  outfile << whereAmI[i].matches << "\t";
										  }
										  outfile << "\n";
										  outfile.close();

										  std::cout << "\nI believe I am at " << locationID << " and " << whereAmI[bestMatch].angle << " off angle!"<< std::endl;
										  switchNum = 0;
										  }*/
					}
					else
					{
						moveCam(0);
						while(!save(feature_sphere));
						std::cout << "File Saved!" << std::endl;
						switchNum = 0;
					}
				}
			}else
				stationary_clock = clock();

		}
            break;

            case 3:
                PTUdif = 0;
                for (int i = 0;i<ptu_state->name.size();i++)
                {
                    if (ptu_state->name[i] == "pan") {
                        ptuAng = ptu_state->position[i] * 60;

                        PTUdif = ptu_state->position[i] > ptuGoal ? ptu_state->position[i] - ptuGoal : ptuGoal - ptu_state->position[i];
                    }
                }
                if(PTUdif < 0.001)
                    switchNum++;

                break;
            case 4:
                locationID = whereAmI[bestIndex[myFundCount]].name;
                finalComparason(conversions(img));
                break;
            case 5:
                myFundCount--;
                if(myFundCount >= 0)
                {
                    //average.clear();
                    //average.push_back(0); average.push_back(0); average.push_back(0);
                    count = 0;
                    subcount = 0;

                    estimatedAngle = whereAmI[bestIndex[myFundCount]].angle;
                    moveCam(((-int(estimatedAngle)-180) % 360)+180);
                    switchNum = 3;
                }else{
                    switchNum = 0;
                    moveCam(0);
		    float avx[NUM_WAYPOINTS];
		    float avy[NUM_WAYPOINTS];
		    float bvx[NUM_WAYPOINTS];
		    float bvy[NUM_WAYPOINTS];
		    float px[NUM_WAYPOINTS];
		    float py[NUM_WAYPOINTS];
		    for (int i = 0;i<NUM_WAYPOINTS;i++)
		    {
			   avx[i] =avy[i] = 0;
			    for (int j = 0;j<NUM_FUNDS;j++)
			    {
			             avx[i] += vx[i][j];
			             avy[i] += vy[i][j];
				    //printf("%i %i %.3f %.3f %.3f %.3f %s\n",i,j,vx[i][j],vy[i][j],whereAmI[bestIndex[i]].angle,whereAmI[bestIndex[i]].angle,whereAmI[bestIndex[i]].name.c_str()); 
			    }
			    avx[i] = avx[i]/NUM_FUNDS;	
			    avy[i] = avy[i]/NUM_FUNDS;
		            bvx[i] = avy[i];
		            bvy[i] = -avx[i];
			    px[i] = read_features[bestIndex[i]].robotPose[0];
			    py[i] = read_features[bestIndex[i]].robotPose[1];	
			    printf("%i %.3f %.3f %.3f %.3f %s\n",i,avx[i],avy[i],px[i],py[i],whereAmI[bestIndex[i]].name.c_str()); 
		    }

		    float c[NUM_WAYPOINTS];
		    for (int i=0;i<NUM_WAYPOINTS;i++) c[i]=px[i]*bvx[i]+bvy[i]*py[i];

		    float a0,a1,a2,b0,b1;
		    a0=a1=a2=b0=b1=0;
		    for (int i=0;i<NUM_WAYPOINTS;i++){
			    a0+= bvx[i]*bvx[i];
			    a1+= bvx[i]*bvy[i];
			    a2+= bvy[i]*bvy[i];
			    b0+= bvx[i]*c[i];
			    b1+= bvy[i]*c[i];
		    }
		    float d=a0*a2-a1*a1;
		    float x =(+b0*a2-b1*a1)/d;
		    float y =(-b0*a1+b1*a0)/d;
		    printf("POS: %.3f %.3f\n",x,y); 
		}
                break;
            default:
                std::cout << "UNKNOWN SWITCH STATEMENT CLOSING" << std::endl;
                std::exit(1);
                break;
            }


        }                // output each degree
        // std::cout << hist[i] << "\t" << roundHist[i] << std::endl;

        void read(std::string fn, bool firstLoop)
        {
            bool isDir = false;
            DIR *dir;
            struct dirent *ent;
            char name[200];
            if ((dir = opendir (fn.c_str())) != NULL && !firstLoop) {
                while ((ent = readdir(dir)) != NULL )
                {
                    dirName = fn;
                    sprintf(name,"%s/%s", fn.c_str(), ent->d_name);
                    read(name, true);// << "\n";
                }
            }
            else
            {
                if(fn.substr(fn.find_last_of(".") + 1) != "yml")
                {
                    return;
                }
                char realFile[200];
                strcpy(realFile, fn.c_str());
                std::ifstream my_file(realFile);
                if (my_file)    // is it a real file
                {
                    feature_struct temp;
                    cv::FileStorage fs2(fn, cv::FileStorage::READ);

                    fs2["descriptors"] >> temp.descriptors; //this->read_features[0].descriptors;

                    cv::FileNode  kptFileNode1 = fs2["keypoints"];

                    cv::read( kptFileNode1, temp.keypoints);// this->read_features[0].keypoints );
                    cv::Mat tempPose(3, 1, CV_64F, robotPose);
                    cv::FileNode  kptFileNode2 = fs2["robotPose"];
                    cv::read( kptFileNode2, tempPose);;
                    temp.robotPose[0] = tempPose.at<double>(0);
                    temp.robotPose[1] = tempPose.at<double>(1);
                    temp.robotPose[2] = tempPose.at<double>(2);
                    fs2.release();

                    this->read_features.push_back(temp);

                    estimation guessTemp;       // reading is likely going to lead to estimations, best handle this here
                    guessTemp.angle = 0;
                    guessTemp.matches = 0;
                    guessTemp.name = fn.substr(fn.find_last_of("/")+1, fn.find_last_of(".") - fn.find_last_of("/")-1);  // trim away directory and extension

                    whereAmI.push_back(guessTemp);
                    std::cout << "Successfully Loaded " << this->read_features[this->read_features.size()-1].keypoints.size() << " keypoints from " << guessTemp.name << std::endl;
                    if(fundChecker)
                        myFundCount++;
                }
            }
        }

        bool CheckCoherentRotation(cv::Mat_<double>& R) {
            if(fabsf(determinant(R))-1.0 > 1e-07) {
                std::cerr << "det(R) != +-1.0, this is not a rotation matrix" << std::endl;
                return false;
            }
            return true;
        }

        void finalComparason(cv::Mat img_2)
        {

            std::cout << "######################## " << subcount << " # " << count << " ########################" << std::endl;
            if(subcount >= 100 && !fundChecker ) // I am clearly lost // perhaps add -> && !fundChecker
            {
                std::cout << "I not am sure where I am so shall create a custom waypoint here!" << std::endl;
                save(feature_sphere);
                int found = 1;
                for(int i =0; i < this->whereAmI.size(); i++)
                {
                    unsigned check_str = whereAmI[i].name.find_last_of("_");
                    if(check_str !=-1)
                        found++;
                }

                std::ostringstream oss;
                oss << dirName << "LostState_" << found;

                locationID = oss.str();

                if(save(feature_sphere))    // save .yml Data
                {
                    ForwardCamFrame;
                    std::stringstream ss;
                    ss<< dirName << "/" << this->locationID<< ".jpg";
                    cv::imwrite( ss.str(), ForwardCamFrame );   // Save frame view forward

                    read(locationID, false);                    // Add to Database
                }

                outfile.open(runOutputFile, std::ios_base::app);
                outfile << headerFormating;
                for(int i = 0; i < whereAmI.size(); i++)
                {
                    outfile << whereAmI[i].name << "\t";
                }
                outfile << "\n";
                outfile.close();

                switchNum = 0;
                return;
            }
            if(count < NUM_FUNDS)
            {
                if(fundChecker)
                    locationID = whereAmI[myFundCount].name;

                std::stringstream ss;
                ss<< dirName << "/" << this->locationID<< ".jpg";
                cv::Mat img_1 = cv::imread( ss.str(), CV_LOAD_IMAGE_GRAYSCALE );


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

                //-- Step 3: Matching descriptor vectors with a brute force matcher
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

                            if(difY < 150 && difY > -150)    // .. this will disregard diagonal matches
                                matches.push_back(match);
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

                //-- Draw matches
                /*cv::Mat img_matches;
                cv::drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches );
                //-- Show detected matches
                cv::namedWindow( "Matches", CV_WINDOW_NORMAL );
                cv::imshow("Matches", img_matches );
                cv::waitKey(1000000);*/


                //-- Step 4: calculate Fundamental Matrix
                std::vector<cv::Point2f>imgpts1,imgpts2;
                for( unsigned int i = 0; i<matches.size(); i++ )
                {

                    imgpts1.push_back(keypoints_1[matches[i].queryIdx].pt);
                    // trainIdx is the "right" image
                    imgpts2.push_back(keypoints_2[matches[i].trainIdx].pt);
                }


                //double data[] = {525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0};    // .. Camera intrinsic matrix from camera_info
                double data[] = {526.60717328, 0.00000000, 318.52510740, 0.00000000, 526.60717328, 241.18145973, 0.00000000, 0.00000000, 1.00000000}; // .. Calibra intrinsic matrix
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


                std::vector<cv::KeyPoint> p1,p2;
                std::vector<CloudPoint> cloud;
                std::vector<CloudPoint> cloud1;

                bool x = FindCameraMatrices(K,Kinv,D,keypoints_1,keypoints_2, p1,p2,P,P1,matches,cloud,cloud1);
		//TOM	
		printf("COUNT: %i %i\n",myFundCount,count);
                vx[myFundCount][count] = P1(0,3);
                vy[myFundCount][count] = P1(2,3);

               // struct CloudPoint {
               //     cv::Point3d pt;
               //     std::vector<int> imgpt_for_img;
               //     double reprojection_error;
               // };

                //std::cout << cloud[0].pt << " " << cloud1[0].pt << std::endl;

                if(x)
                {
                    if(fundChecker)
                    {
                        std::ofstream prnt2File;
                        prnt2File.open("FundAverge.txt", std::ios_base::app);
                        prnt2File << locationID << "\t" << P1(0,0) << "\t" << P1(0,1) << "\t" << P1(0,2) << "\t" << P1(0,3) << "\t" << P1(1,0) << "\t" << P1(1,1) << "\t" << P1(1,2) << "\t" << P1(1,3) << "\t" << P1(2,0) << "\t" << P1(2,1) << "\t" << P1(2,2) << "\t" << P1(2,3) << "\n";
                        prnt2File.close();
                    }

                    average[0] += P1(0,3);
             //       average[1] += P1(1,3);
                    average[2] += P1(2,3);

                    count++;
                }
            }
            else
            {
                std::cout << "My adveraged pose is XY-R: \t" << float(average[0] / count) << "\t" << float(average[1] / count) << "\t"<< float(average[2] / count) << "\t" << estimatedAngle << std::endl;
                std::cout << "\n\nLinda is at: " << locationID << std::endl;

                if(float(average[0] / count) > 0)
                    std::cout << float(average[0] / count) << " m left" << std::endl;
                else
                    std::cout << -float(average[0] / count) << " m right" << std::endl;

                if(float(average[1] / count) < 0)
                    std::cout << -float(average[1] / count) << " ? Foward" << std::endl;
                else
                    std::cout << float(average[1] / count) << " ? Back" << std::endl;

                std::cout << estimatedAngle << " Degrees off recorded data!" << std::endl;

                outfile.open(runOutputFile, std::ios_base::app);
                outfile << groundTruth << locationID << "\t" <<  whereAmI[bestMatch].angle << "\t" << whereAmI[bestMatch].matches << "\t" << float(average[0] / count)  << "\t" << float(average[1] / count)  << "\t";
                for(int i = 0; i < whereAmI.size(); i ++)
                {
                    outfile << whereAmI[i].matches << "\t";
                }
                outfile << "\n";
                outfile.close();

                if(fundChecker)
                    switchNum = 5;
                else
                {
                    switchNum = 0;
                    is_linda_lost = false;
                }
            }
            subcount++;
        }

        void moveCam(float angle)
        {
            std::cout << angle << std::endl;
            ptu.name[0] ="tilt";
            ptu.name[1] ="pan";
            ptu.position[0] = 0.0;
            ptu.position[1] = angle / 60;
            ptuGoal = ptu.position[1];
            ptu.velocity[0] = 2;
            ptu.velocity[1] = 2;
            if(ptu.position[1] <= PTU_UPPERLIMIT && ptu.position[1] >= PTU_LOWERLIMIT)
            {
                ptu_pub.publish(ptu);
            }
            else
            {
                ptu.position[1] = 0;    // reset to nill
                ptu_pub.publish(ptu);
                std::exit(1);
            }
        }

private:
        void poseCallback(const geometry_msgs::Pose::ConstPtr& msg)
        {
            robotPose[0] = msg->position.x ;
            robotPose[1] = msg->position.y;
            robotPose[2] = msg->orientation.z;

            std::string temp = "localised";
            std_msgs::String str;

            if(is_linda_lost)
                temp = "lost";

            str.data = temp;
            isLost.publish(str);
        }

        void extractFeatures(cv::Mat image, feature_struct &sphere, float angle)
        {
            //cv::Rect myROI(264, 0, 112, 480); // Crop image for
            cv::Rect myROI(65, 0, 510, 480); // Crop image for#
            image = image(myROI);

            //std::cout << "sphere type: " << sphere.descriptors.type() << ".\t";
            cv::Mat descriptors;
            std::vector<cv::KeyPoint> keypoints;

            // detector && Extract surf
            detector.detect(image, keypoints);
            extractor.compute(image, keypoints, descriptors);

            cv::KeyPoint temp;
            for(int i = 0; i < keypoints.size(); i++)
            {
                temp = keypoints[i];
                keypoints[i].pt.x  = -atan((temp.pt.x-320)/f) * 180 / PI;
                keypoints[i].pt.y  = -atan((temp.pt.y-240)/f) * 180 / PI;

                keypoints[i].pt.x += angle; // Add PTU angle
            }

            sphere.descriptors.push_back(descriptors);
            sphere.keypoints.insert(sphere.keypoints.end(), keypoints.begin(), keypoints.end() );
        }

        bool save(feature_struct &sphere)
        {
            int dictionarySize=1000;

            cv::TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
            //retries number
            int retries=1;
            //necessary flags
            int flags = cv::KMEANS_PP_CENTERS;
            //Create the BoW (or BoF) trainer
            cv::BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
            cv::Mat output = bowTrainer.cluster(sphere.descriptors);

            cv::Mat old = sphere.descriptors;

            cv::FlannBasedMatcher matcher;
            std::vector< cv::DMatch > matches;
            matcher.match( output, old, matches );

            std::vector<cv::KeyPoint> keypoints;

            for(int i = 0; i < matches.size(); i++)
            {
                keypoints.push_back(sphere.keypoints[matches[i].trainIdx]);
            }

            if(locationID.empty())
            {
                moveCam(0);
                std::cout << "Enter file name to save:\n";
                std::cin >> locationID;
            }

            if(!locationID.empty())
            {
                if(locationID.substr(locationID.find_last_of(".") + 1) != "yml") {
                    locationID.append(".yml");
                }
                cv::FileStorage fs(locationID, cv::FileStorage::WRITE);
                //cv::write(fs, "keypoints", sphere.keypoints);
                //cv::write(fs, "descriptors", sphere.descriptors);

                cv::Mat tempPose(3, 1, CV_64F, robotPose);

                cv::write(fs, "robotPose", tempPose);
                cv::write(fs, "keypoints", keypoints);
                cv::write(fs, "descriptors", output);

                fs.release();

                return true;
            }
            return false;
        }

        void match(feature_struct &a, feature_struct &b)
        {
            float hist[360];
            float roundHist[360];
            for(int i = 0; i < 360; i++)
            {
                hist[i] = 0;
                roundHist[i] = 0;
            }

            //std::vector<std::vector <cv::DMatch> > matches;
            std::vector<std::vector <cv::DMatch> > preProcessedMatches;
            std::vector<cv::DMatch> matches;

            cv::BFMatcher matcher;

            if(a.descriptors.rows > 1 && b.descriptors.rows > 1)
            {
                matcher.knnMatch( a.descriptors, b.descriptors, preProcessedMatches, 2);

                int NumMatches = 0;
                for(   int i=0; i < preProcessedMatches.size(); i++)
                {
                    if (preProcessedMatches[i].size() == 2)
                    {
                        if (preProcessedMatches[i][0].distance < preProcessedMatches[i][1].distance * DISTANCE_FACTOR)
                        {
                            cv::DMatch match = cv::DMatch(preProcessedMatches[i][0].queryIdx, preProcessedMatches[i][0].trainIdx, preProcessedMatches[i][0].distance);

                            float difX = a.keypoints[preProcessedMatches[i][0].queryIdx].pt.x - b.keypoints[preProcessedMatches[i][0].trainIdx].pt.x;
                            float difY = a.keypoints[preProcessedMatches[i][0].queryIdx].pt.y - b.keypoints[preProcessedMatches[i][0].trainIdx].pt.y;
                            float dis = sqrt((difX*difX)+(difY*difY));
                            if(difY < 100 || difY > -100)
                            {
                                matches.push_back(match);
                                NumMatches++;
                            }
                        }else{
                            cv::DMatch match = cv::DMatch(preProcessedMatches[i][0].queryIdx, -1, preProcessedMatches[i][0].distance);
                            matches.push_back(match);
                        }
                    }

                    if( matches[i].trainIdx > -1)
                    {
                        // NORMAL CODE
                        int difAng = ceil(a.keypoints[preProcessedMatches[i][0].queryIdx].pt.x - b.keypoints[preProcessedMatches[i][0].trainIdx].pt.x);
                        if(difAng < 0)
                            difAng = 360 + difAng;// + round(realRobot_pose);

                        //hist[difAng]++;

                        {
                            hist[difAng] += 1;
                        }
                    }
                }

                if(NumMatches > 0)
                {
                    int bestPointer = 0;
                    float angle = 0;
                    std::cout << std::endl;
                    for(int i = 0; i < 360; i++)
                    {
                        myfile << hist[i] << "\t";
                        for(int j = 0; j < 5; j++)
                        {
                            roundHist[i] += hist[(i-2+j)%360];
                        }
                        //roundHist[i] /= 5;
                        if(bestPointer < hist[i])
                        {
                            bestPointer=hist[i];
                            angle = (float)i;
                        }
                        // output each degree
                       // std::cout << hist[i] << "\t" << roundHist[i] << std::endl;
                    }
                    //std::cout << std::endl;
                    std::cout << "Estimation for: " << whereAmI[pointer].name << " " << angle << " degrees, (" << NumMatches << " Matches, " << bestPointer << " Peak)." << std::endl;

                    whereAmI[this->pointer].angle = angle;
                    whereAmI[this->pointer].matches = NumMatches;
                }
                else
                {
                    std::cout << "Too Few matches" << std::endl;

                    whereAmI[this->pointer].angle = 0;
                    whereAmI[this->pointer].matches = -1;
                }
            }
        }

protected:  // .. This is why they made header files
        ros::NodeHandle n;
        bool is_linda_lost;
        int count, subcount;
        unsigned int pointer;
        float f;
        cv::SurfFeatureDetector detector;
        cv::SurfDescriptorExtractor extractor;
        cv::FlannBasedMatcher matcher;
        feature_struct feature_sphere;

        float ptu_start_angle;
        float ptu_angle;

        std::ofstream myfile;

        // new
        ros::Publisher  ptu_pub;
        ros::Publisher  isLost;
        ros::Subscriber pose_sub;
        ros::Subscriber lost_sub;
        sensor_msgs::JointState ptu;
        ros::ServiceServer service;
        float lastPtuAng, ptuAng;
        float ptu_pose;
        float ptuGoal;
        int ptu_iterator; // in degrees
        float ptu_stationary;
        clock_t stationary_clock;
        float realRobot_pose;
        std::string dirName;
        std::vector<feature_struct> read_features;
        int estimatedAngle;

        cv::Mat ForwardCamFrame;
        cv::Ptr<cv::DescriptorMatcher> matcher2;
        cv::Ptr<cv::DescriptorExtractor> extractor2;
        cv::Ptr<cv::BOWImgDescriptorExtractor> dextract;
        cv::SurfFeatureDetector detector2();
        cv::Ptr<cv::BOWKMeansTrainer> bow;
        std::vector<double > average;
        int bestMatch;
};

bool reset(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
{
    std::cout << "reset" << std::endl;

    return true;
}

bool estLocation(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
{
    std::cout << "estLocation" << std::endl;

    return true;
}

bool recovery(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
{
    std::cout << "recovery" << std::endl;

    return true;
}

int main (int argc, char** argv)
{
    ros::init(argc, argv, "topological_nav");
    std::cout << "running" << std::endl;
    ros::NodeHandle n;
    


    ptu_features camClass(&n);

    bool getData = false;    // true for runs

    // read old location Data
    if(argc > 1)
        if(!getData)
        {
            camClass.read(argv[1], false);
            if(camClass.whereAmI.size() > 0)
            {
                camClass.outfile.open(runOutputFile, std::ios_base::app);
                camClass.outfile << headerFormating;
                for(int i = 0; i < camClass.whereAmI.size(); i++)
                {
                    camClass.outfile << camClass.whereAmI[i].name << "\t";
                }
                camClass.outfile << "\n";
                camClass.outfile.close();
            }
        }

    message_filters::Subscriber<sensor_msgs::Image> image_sub(n, "/head_xtion/rgb/image_color", 1);
    message_filters::Subscriber<sensor_msgs::JointState> ptu_sub(n, "/ptu/state", 1);
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(33), image_sub, ptu_sub);

    sync.registerCallback(boost::bind(&ptu_features::callback, &camClass, _1, _2));

    ros::spin();

    return 0;
}

