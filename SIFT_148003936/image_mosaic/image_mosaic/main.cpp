
#include "opencv2\opencv.hpp"

extern "C"

{

#include "sift.h"
#include "imgfeatures.h"
#include "kdtree.h"
#include"minpq.h"
#include"utils.h"
#include"xform.h"

}
using namespace std;
using namespace cv;
#define KDTREE_BBF_MAX_NN_CHKS 200 //��BBF�㷨��������ڽ���������������
#define NN_SQ_DIST_RATIO_THR 0.6 //Ϊ����ڵ�ʹν��ڵ�����ƽ������ķ�ֵ1
void CalcFourCorner(CvMat *H, Point *leftTop, Point *leftBottom,Point *rightTop,Point *rightBottom, IplImage *img2)
{
	//����ͼ2���ĸ��Ǿ�����H�任�������
	double v2[] = { 0,0,1 };//���Ͻ�
	double v1[3];//�任�������ֵ
	CvMat V2 = cvMat(3, 1, CV_64FC1, v2);
	CvMat V1 = cvMat(3, 1, CV_64FC1, v1);
	cvGEMM(H, &V2, 1, 0, 1, &V1);//����˷�
	leftTop->x = cvRound(v1[0] / v1[2]);
	leftTop->y = cvRound(v1[1] / v1[2]);
	//cvCircle(xformed,leftTop,7,CV_RGB(255,0,0),2);

	//��v2��������Ϊ���½�����
	v2[0] = 0;
	v2[1] = img2->height;
	V2 = cvMat(3, 1, CV_64FC1, v2);
	V1 = cvMat(3, 1, CV_64FC1, v1);
	cvGEMM(H, &V2, 1, 0, 1, &V1);
	leftBottom->x = cvRound(v1[0] / v1[2]);
	leftBottom->y = cvRound(v1[1] / v1[2]);
	//cvCircle(xformed,leftBottom,7,CV_RGB(255,0,0),2);

	//��v2��������Ϊ���Ͻ�����
	v2[0] = img2->width;
	v2[1] = 0;
	V2 = cvMat(3, 1, CV_64FC1, v2);
	V1 = cvMat(3, 1, CV_64FC1, v1);
	cvGEMM(H, &V2, 1, 0, 1, &V1);
	rightTop->x = cvRound(v1[0] / v1[2]);
	rightTop->y = cvRound(v1[1] / v1[2]);
	//cvCircle(xformed,rightTop,7,CV_RGB(255,0,0),2);

	//��v2��������Ϊ���½�����
	v2[0] = img2->width;
	v2[1] = img2->height;
	V2 = cvMat(3, 1, CV_64FC1, v2);
	V1 = cvMat(3, 1, CV_64FC1, v1);
	cvGEMM(H, &V2, 1, 0, 1, &V1);
	rightBottom->x = cvRound(v1[0] / v1[2]);
	rightBottom->y = cvRound(v1[1] / v1[2]);
	//cvCircle(xformed,rightBottom,7,CV_RGB(255,0,0),2);
}
int main()
{
	IplImage *img1, *img2, *img1_Feat, *img2_Feat;
	struct feature *feat1, *feat2;
	int n1, n2;
	string path_img1 = "C:/Users/ZYX/OneDrive/vr����ҵ/sift/ͼ��ƴ��/test/90.jpg";
	string path_img2 = "C:/Users/ZYX/OneDrive/vr����ҵ/sift/ͼ��ƴ��/test/89.jpg";
	string path_img = "C:/Users/ZYX/OneDrive/vr����ҵ/sift/ͼ��ƴ��/answer.jpg";
	img1 = cvLoadImage(path_img1.c_str());
	img2 = cvLoadImage(path_img2.c_str());
	//
	//cvSetImageROI(img1, cvRect(img1->width - 500, 0, 500, img1->height));
	//IplImage *img1_right= cvCreateImage(cvSize(500,img1->height), img1->depth, img1->nChannels);
	//cvCopy(img1,img1_right);
	//cvResetImageROI(img1);
	//img1 = img1_right;
	//
	img1_Feat = cvCloneImage(img1);//����ͼ1�������������������  
	img2_Feat = cvCloneImage(img2);//����ͼ2�������������������  
	

								   //Ĭ����ȡ����LOWE��ʽ��SIFT������  
								   //��ȡ����ʾ��1��ͼƬ�ϵ�������  
	n1 = sift_features(img1, &feat1);//���ͼ1�е�SIFT������,n1��ͼ1�����������  
	export_features("feature1.txt", feat1, n1);//��������������д�뵽�ļ�  
	draw_features(img1_Feat, feat1, n1);//����������  
	//cvNamedWindow("IMG1_FEAT");//��������  
	//cvShowImage("IMG1_FEAT", img1_Feat);//��ʾ  

	cvWaitKey(0);
	cvSaveImage("C:/Users/ZYX/OneDrive/vr����ҵ/sift/ͼ��ƴ��/test1/sift1.jpg", img1_Feat);
	//��ȡ����ʾ��2��ͼƬ�ϵ�������  
	n2 = sift_features(img2, &feat2);//���ͼ2�е�SIFT�����㣬n2��ͼ2�����������  
	export_features("feature2.txt", feat2, n2);//��������������д�뵽�ļ�  
	draw_features(img2_Feat, feat2, n2);//����������  
	//cvNamedWindow("IMG2_FEAT");//��������  
	//cvShowImage("IMG2_FEAT", img2_Feat);//��ʾ  
	//cvWaitKey(0);
	cvSaveImage("C:/Users/ZYX/OneDrive/vr����ҵ/sift/ͼ��ƴ��/test1/sift2.jpg", img2_Feat);
	/////////////////////////////////////////////////////////
	struct kd_node *kd_root;
	struct feature *feat;
	struct feature** nbrs;
	IplImage *stacked = stack_imgs1(img1, img2);
	//����ͼ1�������㼯feat1����k-d��������k-d������kd_root  
	kd_root = kdtree_build(feat1, n1);
	Point pt1, pt2;//���ߵ������˵�  
	double d0, d1;//feat2��ÿ�������㵽����ںʹν��ڵľ���  
	int matchNum = 0;//�������ֵ��ɸѡ���ƥ���Եĸ���  
		//���������㼯feat2�����feat2��ÿ��������feat��ѡȡ���Ͼ����ֵ������ƥ��㣬�ŵ�feat��fwd_match����  
	for (int i = 0; i < n2; i++)
	{
		feat = feat2 + i;//��i���������ָ��  
		//��kd_root������Ŀ���feat��2������ڵ㣬�����nbrs�У�����ʵ���ҵ��Ľ��ڵ����  
		int k = kdtree_bbf_knn(kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS);
		if (k == 2)
		{
			d0 = descr_dist_sq(feat, nbrs[0]);//feat������ڵ�ľ����ƽ��  
			d1 = descr_dist_sq(feat, nbrs[1]);//feat��ν��ڵ�ľ����ƽ��  
			//��d0��d1�ı�ֵС����ֵNN_SQ_DIST_RATIO_THR������ܴ�ƥ�䣬�����޳�  
			if (d0 < d1 * NN_SQ_DIST_RATIO_THR)
			{   //��Ŀ���feat������ڵ���Ϊƥ����  
				pt2 = Point(cvRound(feat->x), cvRound(feat->y));//ͼ2�е������  
				pt1 = Point(cvRound(nbrs[0]->x), cvRound(nbrs[0]->y));//ͼ1�е������(feat������ڵ�)  
				pt2.x += img1->width;//��������ͼ���������еģ�pt2�ĺ��������ͼ1�Ŀ�ȣ���Ϊ���ߵ��յ�  
				cvLine(stacked, pt1, pt2, CV_RGB(255, 0, 255), 1, 8, 0);//��������  
				matchNum++;//ͳ��ƥ���Եĸ���  
				feat2[i].fwd_match = nbrs[0];//ʹ��feat��fwd_match��ָ�����Ӧ��ƥ���  
			}
		}
		free(nbrs);//�ͷŽ�������  
	}
	//��ʾ�����澭�����ֵ��ɸѡ���ƥ��ͼ  
	//cvNamedWindow("IMG_MATCH1");//��������  
	//cvShowImage("IMG_MATCH1", stacked);//��ʾ  
	//cvWaitKey(0);
	//cvSaveImage("C:/Users/ZYX/OneDrive/vr����ҵ/sift/ͼ��ƴ��/test1/match1.jpg", stacked);
	/////////////////////////////////////////////////////////////
	struct feature **inliers;
	int n_inliers;
	CvMat *H;
	IplImage *stacked_ransac = stack_imgs1(img1, img2);
	//����RANSAC�㷨ɸѡƥ���,����任����H��  
	//����img1��img2������˳�򣬼������H��Զ�ǽ�feat2�е�������任Ϊ��ƥ��㣬����img2�еĵ�任Ϊimg1�еĶ�Ӧ��  
	H = ransac_xform(feat2, n2, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01, homog_xfer_err, 3.0, &inliers, &n_inliers);

	//���ܳɹ�������任���󣬼�����ͼ���й�ͬ����  
	if (H)
	{
		cout << "��RANSAC�㷨ɸѡ���ƥ���Ը�����" << n_inliers << endl; //���ɸѡ���ƥ���Ը���  

		int invertNum = 0;//ͳ��pt2.x > pt1.x��ƥ���Եĸ��������ж�img1���Ƿ���ͼ  

						  //������RANSAC�㷨ɸѡ��������㼯��inliers���ҵ�ÿ���������ƥ��㣬��������  
		for (int i = 0; i < n_inliers; i++)
		{
			feat = inliers[i];//��i��������  
			pt2 = Point(cvRound(feat->x), cvRound(feat->y));//ͼ2�е������  
			pt1 = Point(cvRound(feat->fwd_match->x), cvRound(feat->fwd_match->y));//ͼ1�е������(feat��ƥ���)  

																				  //ͳ��ƥ��������λ�ù�ϵ�����ж�ͼ1��ͼ2������λ�ù�ϵ  
			if (pt2.x > pt1.x)
				invertNum++;

			pt2.x += img1->width;//��������ͼ���������еģ�pt2�ĺ��������ͼ1�Ŀ�ȣ���Ϊ���ߵ��յ�  
			cvLine(stacked_ransac, pt1, pt2, CV_RGB(255, 0, 255), 1, 8, 0);//��ƥ��ͼ�ϻ�������  
		}

		//cvNamedWindow("IMG_MATCH2");//��������  
		//cvShowImage("IMG_MATCH2", stacked_ransac);//��ʾ��RANSAC�㷨ɸѡ���ƥ��ͼ  

												/*�����м�����ı任����H������img2�еĵ�任Ϊimg1�еĵ㣬���������img1Ӧ������ͼ��img2Ӧ������ͼ��
												��ʱimg2�еĵ�pt2��img1�еĶ�Ӧ��pt1��x����Ĺ�ϵ�������ǣ�pt2.x < pt1.x
												���û��򿪵�img1����ͼ��img2����ͼ����img2�еĵ�pt2��img1�еĶ�Ӧ��pt1��x����Ĺ�ϵ�������ǣ�pt2.x > pt1.x
												����ͨ��ͳ�ƶ�Ӧ��任ǰ��x�����С��ϵ������֪��img1�ǲ�����ͼ��
												���img1����ͼ����img1�е�ƥ��㾭H������H_IVT�任��ɵõ�img2�е�ƥ���*/

												//��pt2.x > pt1.x�ĵ�ĸ��������ڵ������80%�����϶�img1������ͼ  
		//cvWaitKey(0);
		//cvSaveImage("C:/Users/ZYX/OneDrive/vr����ҵ/sift/ͼ��ƴ��/test1/match2.jpg", stacked_ransac);
		if (invertNum > n_inliers * 0.8)
		{
			CvMat * H_IVT = cvCreateMat(3, 3, CV_64FC1);//�任����������  
														//��H������H_IVTʱ�����ɹ���������ط���ֵ  
			if (cvInvert(H, H_IVT))
			{
				cvReleaseMat(&H);//�ͷű任����H����Ϊ�ò�����  
				H = cvCloneMat(H_IVT);//��H������H_IVT�е����ݿ�����H��  
				cvReleaseMat(&H_IVT);//�ͷ�����H_IVT  
									 //��img1��img2�Ե�  
				IplImage * temp = img2;
				img2 = img1;
				img1 = temp;
			}
			else//H������ʱ������0  
			{
				cvReleaseMat(&H_IVT);//�ͷ�����H_IVT  
				cout << "�任����H������" << endl;
			}
		}
	}
	else //�޷�������任���󣬼�����ͼ��û���غ�����  
	{
		cout << "Warning: ��ͼ���޹�������";
	}

	/////////////////////////////////////////////
	IplImage *xformed, *xformed_simple, *xformed_proc;
	Point leftTop, leftBottom, rightTop, rightBottom;
	//���ܳɹ�������任���󣬼�����ͼ���й�ͬ���򣬲ſ��Խ���ȫ��ƴ��  
	if (H)
	{
		//ƴ��ͼ��img1����ͼ��img2����ͼ  
		CalcFourCorner(H, &leftTop, &leftBottom, &rightTop, &rightBottom, img2);//����ͼ2���ĸ��Ǿ��任�������  
						 //Ϊƴ�ӽ��ͼxformed����ռ�,�߶�Ϊͼ1ͼ2�߶ȵĽ�С�ߣ�����ͼ2���ϽǺ����½Ǳ任��ĵ��λ�þ���ƴ��ͼ�Ŀ��
	
		xformed = cvCreateImage(cvSize(MIN(rightTop.x, rightBottom.x), MIN(img1->height, img2->height)), IPL_DEPTH_8U, 3);
		//�ñ任����H����ͼimg2��ͶӰ�任(�任�������������)������ŵ�xformed��  
		cvWarpPerspective(img2, xformed, H, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
		cvNamedWindow("IMG_MOSAIC_TEMP"); //��ʾ��ʱͼ,��ֻ��ͼ2�任���ͼ  
		cvShowImage("IMG_MOSAIC_TEMP", xformed);
		cvWaitKey(0);
		//����ƴ�ӷ���ֱ�ӽ�����ͼimg1���ӵ�xformed�����  
		xformed_simple = cvCloneImage(xformed);//����ƴ��ͼ����¡��xformed  
		cvSetImageROI(xformed_simple, cvRect(0, 0, img1->width, img1->height));
		cvAddWeighted(img1, 1, xformed_simple, 0, 0, xformed_simple);
		cvResetImageROI(xformed_simple);
		//cvNamedWindow("IMG_MOSAIC_SIMPLE");//��������  
		//cvShowImage("IMG_MOSAIC_SIMPLE", xformed_simple);//��ʾ����ƴ��ͼ  
		//cvWaitKey(0);
		//������ƴ��ͼ����¡��xformed  
		xformed_proc = cvCloneImage(xformed);

		//�ص�������ߵĲ�����ȫȡ��ͼ1  
		cvSetImageROI(img1, cvRect(0, 0, MAX(0,MIN(leftTop.x, leftBottom.x)), xformed_proc->height));
		cvSetImageROI(xformed, cvRect(0, 0, MAX(0,MIN(leftTop.x, leftBottom.x)), xformed_proc->height));
		cvSetImageROI(xformed_proc, cvRect(0, 0, MAX(0,MIN(leftTop.x, leftBottom.x)), xformed_proc->height));
		cvAddWeighted(img1, 1, xformed, 0, 0, xformed_proc);
		cvResetImageROI(img1);
		cvResetImageROI(xformed);
		cvResetImageROI(xformed_proc);
		//cvNamedWindow("IMG_MOSAIC_BEFORE_FUSION");
		//cvShowImage("IMG_MOSAIC_BEFORE_FUSION", xformed_proc);//��ʾ�ں�֮ǰ��ƴ��ͼ  
		//cvWaitKey(0);
		//���ü�Ȩƽ���ķ����ں��ص�����  
		int start = MAX(0,MIN(leftTop.x, leftBottom.x));//��ʼλ�ã����ص��������߽�  
		double processWidth = img1->width - start;//�ص�����Ŀ��  
		double alpha = 1;//img1�����ص�Ȩ��  
		for (int i = 0; i < xformed_proc->height; i++)//������  
		{
			const uchar * pixel_img1 = ((uchar *)(img1->imageData + img1->widthStep * i));//img1�е�i�����ݵ�ָ��  
			const uchar * pixel_xformed = ((uchar *)(xformed->imageData + xformed->widthStep * i));//xformed�е�i�����ݵ�ָ��  
			uchar * pixel_xformed_proc = ((uchar *)(xformed_proc->imageData + xformed_proc->widthStep * i));//xformed_proc�е�i�����ݵ�ָ��  
			for (int j = start; j < img1->width; j++)//�����ص��������  
			{
				//�������ͼ��xformed�������صĺڵ㣬����ȫ����ͼ1�е�����  
				if (pixel_xformed[j * 3] < 50 && pixel_xformed[j * 3 + 1] < 50 && pixel_xformed[j * 3 + 2] < 50)
				{
					alpha = 1;
				}
				else
				{   //img1�����ص�Ȩ�أ��뵱ǰ�������ص�������߽�ľ��������  
					alpha = (processWidth - (j - start)) / processWidth;
				}
				pixel_xformed_proc[j * 3] = pixel_img1[j * 3] * alpha + pixel_xformed[j * 3] * (1 - alpha);//Bͨ��  
				pixel_xformed_proc[j * 3 + 1] = pixel_img1[j * 3 + 1] * alpha + pixel_xformed[j * 3 + 1] * (1 - alpha);//Gͨ��  
				pixel_xformed_proc[j * 3 + 2] = pixel_img1[j * 3 + 2] * alpha + pixel_xformed[j * 3 + 2] * (1 - alpha);//Rͨ��  
			}
		}
		//cvNamedWindow("IMG_MOSAIC_PROC");//��������  
		//cvShowImage("IMG_MOSAIC_PROC", xformed_proc);//��ʾ������ƴ��ͼ  
		//cvWaitKey(0);
		cvSaveImage(path_img.c_str(), xformed_proc);
	
		return 0;
	}
}