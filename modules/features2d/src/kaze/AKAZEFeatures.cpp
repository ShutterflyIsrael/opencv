/**
 * @file AKAZEFeatures.cpp
 * @brief Main class for detecting and describing binary features in an
 * accelerated nonlinear scale space
 * @date Sep 15, 2013
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */


#include "../precomp.hpp"
#include "AKAZEFeatures.h"
#include "fed.h"
#include "nldiffusion_functions.h"
#include "utils.h"

#include <iostream>

// Namespaces
namespace cv
{
using namespace std;

/* ************************************************************************* */
/**
 * @brief AKAZEFeatures constructor with input options
 * @param options AKAZEFeatures configuration options
 * @note This constructor allocates memory for the nonlinear scale space
 */
AKAZEFeatures::AKAZEFeatures(const AKAZEOptions& options) : options_(options) {

  ncycles_ = 0;
  reordering_ = true;

  if (options_.descriptor_size > 0 && options_.descriptor >= AKAZE::DESCRIPTOR_MLDB_UPRIGHT) {
    generateDescriptorSubsample(descriptorSamples_, descriptorBits_, options_.descriptor_size,
                                options_.descriptor_pattern_size, options_.descriptor_channels);
  }

  Allocate_Memory_Evolution();
}

/* ************************************************************************* */
/**
 * @brief This method allocates the memory for the nonlinear diffusion evolution
 */
void AKAZEFeatures::Allocate_Memory_Evolution(void) {

  float rfactor = 0.0f;
  int level_height = 0, level_width = 0;

  // Allocate the dimension of the matrices for the evolution
  for (int i = 0, power = 1; i <= options_.omax - 1; i++, power *= 2) {
    rfactor = 1.0f / power;
    level_height = (int)(options_.img_height*rfactor);
    level_width = (int)(options_.img_width*rfactor);

    // Smallest possible octave and allow one scale if the image is small
    if ((level_width < 24 || level_height < 24) && i != 0) {
      options_.omax = i;
      break;
    }

    for (int j = 0; j < options_.nsublevels; j++) {
      TEvolution step;
      step.Lx = Mat::zeros(level_height, level_width, CV_32F);
      step.Ly = Mat::zeros(level_height, level_width, CV_32F);
      step.Lxx = Mat::zeros(level_height, level_width, CV_32F);
      step.Lxy = Mat::zeros(level_height, level_width, CV_32F);
      step.Lyy = Mat::zeros(level_height, level_width, CV_32F);
      step.Lt = Mat::zeros(level_height, level_width, CV_32F);
      step.Ldet = Mat::zeros(level_height, level_width, CV_32F);
      step.Lsmooth = Mat::zeros(level_height, level_width, CV_32F);
      step.esigma = options_.soffset*pow(2.f, (float)(j) / (float)(options_.nsublevels) + i);
      step.sigma_size = fRound(step.esigma);
      step.etime = 0.5f*(step.esigma*step.esigma);
      step.octave = i;
      step.sublevel = j;
      evolution_.push_back(step);
    }
  }

  // Allocate memory for the number of cycles and time steps
  for (size_t i = 1; i < evolution_.size(); i++) {
    int naux = 0;
    vector<float> tau;
    float ttime = 0.0f;
    ttime = evolution_[i].etime - evolution_[i - 1].etime;
    naux = fed_tau_by_process_time(ttime, 1, 0.25f, reordering_, tau);
    nsteps_.push_back(naux);
    tsteps_.push_back(tau);
    ncycles_++;
  }
}

/* ************************************************************************* */
/**
 * @brief This method creates the nonlinear scale space for a given image
 * @param img Input image for which the nonlinear scale space needs to be created
 * @return 0 if the nonlinear scale space was created successfully, -1 otherwise
 */


 
 
int AKAZEFeatures::Create_Nonlinear_Scale_Space(const Mat& img)
{
  CV_Assert(evolution_.size() > 0);
 
 img.copyTo(evolution_[0].Lt);

  gaussian_2D_convolution(evolution_[0].Lt, evolution_[0].Lt, 0, 0, options_.soffset);//2.14
  evolution_[0].Lt.copyTo(evolution_[0].Lsmooth);
  
  // Allocate memory for the flow and step images
  Mat Lflow = Mat(evolution_[0].Lt.rows, evolution_[0].Lt.cols, CV_32F);
  Mat Lstep = Mat(evolution_[0].Lt.rows, evolution_[0].Lt.cols, CV_32F);

 
  // First compute the kcontrast factor
 
  //options_.kcontrast = compute_k_percentile(img, options_.kcontrast_percentile, 1.0f, options_.kcontrast_nbins, 0, 0);//10
  
	
   
  options_.kcontrast=.01;
  // Now generate the rest of evolution levels
  for (size_t i = 1; i < evolution_.size(); i++)
 {
 
    if (evolution_[i].octave > evolution_[i - 1].octave)
	{ 

      halfsample_image(evolution_[i - 1].Lt, evolution_[i].Lt);//12.7
	  options_.kcontrast = options_.kcontrast*0.75f;
	
      // Allocate memory for the resized flow and step images
	
     Lflow = Mat(evolution_[i].Lt.rows, evolution_[i].Lt.cols, CV_32F);//2.3
     Lstep = Mat(evolution_[i].Lt.rows, evolution_[i].Lt.cols, CV_32F);
	
    }
    else
	{
   
      evolution_[i - 1].Lt.copyTo(evolution_[i].Lt); // 3.6
	
    }

  gaussian_2D_convolution(evolution_[i].Lt, evolution_[i].Lsmooth, 0, 0, 1.0f);//12.5//5.5
  image_derivatives_scharr(evolution_[i].Lsmooth, evolution_[i].Lx, 1, 0);//7.15
  image_derivatives_scharr(evolution_[i].Lsmooth, evolution_[i].Ly, 0, 1);

          
    switch (options_.diffusivity)
	{
      case KAZE::DIFF_PM_G1:
        pm_g1(evolution_[i].Lx, evolution_[i].Ly, Lflow, options_.kcontrast);
      break;
      case KAZE::DIFF_PM_G2:
        pm_g2(evolution_[i].Lx, evolution_[i].Ly, Lflow, options_.kcontrast);
      break;
      case KAZE::DIFF_WEICKERT:
        weickert_diffusivity(evolution_[i].Lx, evolution_[i].Ly, Lflow, options_.kcontrast);
      break;
      case KAZE::DIFF_CHARBONNIER:
        charbonnier_diffusivity(evolution_[i].Lx, evolution_[i].Ly, Lflow, options_.kcontrast);//2.6
      break;
      default:
        CV_Error(options_.diffusivity, "Diffusivity is not supported");
      break;
    }
	
    // Perform FED n inner steps
    for (int j = 0; j < nsteps_[i - 1]; j++) 
	{
      nld_step_scalar(evolution_[i].Lt, Lflow, Lstep, tsteps_[i - 1][j]);//50//16
    }
	;
  }

 
  return 0;
}


/* ************************************************************************* */
/**
 * @brief This method selects interesting keypoints through the nonlinear scale space
 * @param kpts Vector of detected keypoints
 */
void AKAZEFeatures::Feature_Detection(std::vector<KeyPoint>& kpts)
{
 

  kpts.clear();
  Compute_Determinant_Hessian_Response();
   
  Find_Scale_Space_Extrema(kpts);
  Do_Subpixel_Refinement(kpts);
 

}

/* ************************************************************************* */
class MultiscaleDerivativesAKAZEInvoker : public ParallelLoopBody
{
public:
    explicit MultiscaleDerivativesAKAZEInvoker(std::vector<TEvolution>& ev, const AKAZEOptions& opt)
    : evolution_(&ev)
    , options_(opt)
  {
  }

  void operator()(const Range& range) const
  {
    std::vector<TEvolution>& evolution = *evolution_;

    for (int i = range.start; i < range.end; i++)
    {
        float ratio = (float)fastpow(2, evolution[i].octave);
      int sigma_size_ = fRound(evolution[i].esigma * options_.derivative_factor / ratio);
	 
     compute_scharr_derivatives(evolution[i].Lsmooth, evolution[i].Lx, 1, 0, sigma_size_);
     compute_scharr_derivatives(evolution[i].Lsmooth, evolution[i].Ly, 0, 1, sigma_size_);
	
     compute_scharr_derivatives(evolution[i].Lx, evolution[i].Lxx, 1, 0, sigma_size_);
      compute_scharr_derivatives(evolution[i].Ly, evolution[i].Lyy, 0, 1, sigma_size_);
      compute_scharr_derivatives(evolution[i].Lx, evolution[i].Lxy, 0, 1, sigma_size_);
	
     evolution[i].Lx = evolution[i].Lx*((sigma_size_));
     evolution[i].Ly = evolution[i].Ly*((sigma_size_));
      evolution[i].Lxx = evolution[i].Lxx*((sigma_size_)*(sigma_size_));
      evolution[i].Lxy = evolution[i].Lxy*((sigma_size_)*(sigma_size_));
     evolution[i].Lyy = evolution[i].Lyy*((sigma_size_)*(sigma_size_));
    }
  }

private:
  std::vector<TEvolution>*  evolution_;
  AKAZEOptions              options_;
};

/* ************************************************************************* */
/**
 * @brief This method computes the multiscale derivatives for the nonlinear scale space
 */
void AKAZEFeatures::Compute_Multiscale_Derivatives(void)
{
  parallel_for_(Range(0, (int)evolution_.size()),
                                        MultiscaleDerivativesAKAZEInvoker(evolution_, options_));
}

/* ************************************************************************* */
/**
 * @brief This method computes the feature detector response for the nonlinear scale space
 * @note We use the Hessian determinant as the feature detector response
 */
void AKAZEFeatures::Compute_Determinant_Hessian_Response(void)
{

  // Firstly compute the multiscale derivatives
 
	Compute_Multiscale_Derivatives();
 
  for (size_t i = 0; i < evolution_.size(); i++)
  {  
	/*  float ratio = (float)fastpow(2, evolution_[i].octave);
      int sigma_size_ = fRound(evolution_[i].esigma * options_.derivative_factor / ratio);
	  image_derivatives_scharr_asm(evolution_[i].Lsmooth,evolution_[i].Lx ,evolution_[i].Ly ,sigma_size_);
	    image_derivatives_scharr_asm(evolution_[i].Ly,evolution_[i].Lxy ,evolution_[i].Lyy ,sigma_size_);
	   image_derivatives_scharr_asm(evolution_[i].Lx,evolution_[i].Lxx ,evolution_[i].Lxy ,sigma_size_);
     */
    for (int ix = 0; ix < evolution_[i].Ldet.rows; ix++)
    {
      for (int jx = 0; jx < evolution_[i].Ldet.cols; jx++)
      {
        float lxx = *(evolution_[i].Lxx.ptr<float>(ix)+jx);
        float lxy = *(evolution_[i].Lxy.ptr<float>(ix)+jx);
        float lyy = *(evolution_[i].Lyy.ptr<float>(ix)+jx);
        *(evolution_[i].Ldet.ptr<float>(ix)+jx) = (lxx*lyy - lxy*lxy);
      }
    }
  }
 
}

/* ************************************************************************* */
/**
 * @brief This method finds extrema in the nonlinear scale space
 * @param kpts Vector of detected keypoints
 */
void AKAZEFeatures::Find_Scale_Space_Extrema(std::vector<KeyPoint>& kpts)
{

  float value = 0.0;
  float dist = 0.0, ratio = 0.0, smax = 0.0;
  int npoints = 0, id_repeated = 0;
  int sigma_size_ = 0, left_x = 0, right_x = 0, up_y = 0, down_y = 0;
  bool is_extremum = false, is_repeated = false, is_out = false;
  KeyPoint point;
  vector<KeyPoint> kpts_aux;

  // Set maximum size
  if (options_.descriptor == AKAZE::DESCRIPTOR_MLDB_UPRIGHT || options_.descriptor == AKAZE::DESCRIPTOR_MLDB) {
    smax = 10.0f*sqrtf(2.0f);
  }
  else if (options_.descriptor == AKAZE::DESCRIPTOR_KAZE_UPRIGHT || options_.descriptor == AKAZE::DESCRIPTOR_KAZE) {
    smax = 12.0f*sqrtf(2.0f);
  }

  for (size_t i = 0; i < evolution_.size(); i++) {
    float* prev = evolution_[i].Ldet.ptr<float>(0);
    float* curr = evolution_[i].Ldet.ptr<float>(1);
    for (int ix = 1; ix < evolution_[i].Ldet.rows - 1; ix++) {
      float* next = evolution_[i].Ldet.ptr<float>(ix + 1);

      for (int jx = 1; jx < evolution_[i].Ldet.cols - 1; jx++) {
        is_extremum = false;
        is_repeated = false;
        is_out = false;
        value = *(evolution_[i].Ldet.ptr<float>(ix)+jx);

        // Filter the points with the detector threshold
        if (value > options_.dthreshold && value >= options_.min_dthreshold &&
            value > curr[jx-1] &&
            value > curr[jx+1] &&
            value > prev[jx-1] &&
            value > prev[jx] &&
            value > prev[jx+1] &&
            value > next[jx-1] &&
            value > next[jx] &&
            value > next[jx+1]) {

          is_extremum = true;
          point.response = fabs(value);
          point.size = evolution_[i].esigma*options_.derivative_factor;
          point.octave = (int)evolution_[i].octave;
          point.class_id = (int)i;
          ratio = (float)fastpow(2, point.octave);
          sigma_size_ = fRound(point.size / ratio);
        
          point.pt.x = static_cast<float>(jx);
          point.pt.y = static_cast<float>(ix);

          // Compare response with the same and lower scale
          for (size_t ik = 0; ik < kpts_aux.size(); ik++) {

            if ((point.class_id - 1) == kpts_aux[ik].class_id ||
                point.class_id == kpts_aux[ik].class_id) {
              float distx = point.pt.x*ratio - kpts_aux[ik].pt.x;
              float disty = point.pt.y*ratio - kpts_aux[ik].pt.y;
              dist = distx * distx + disty * disty;
              if (dist <= point.size * point.size) {
                if (point.response > kpts_aux[ik].response) {
                  id_repeated = (int)ik;
                  is_repeated = true;
                }
                else {
                  is_extremum = false;
                }
                break;
              }
            }
          }

          // Check out of bounds
          if (is_extremum == true) {

            // Check that the point is under the image limits for the descriptor computation
            left_x = fRound(point.pt.x - smax*sigma_size_) - 1;
            right_x = fRound(point.pt.x + smax*sigma_size_) + 1;
            up_y = fRound(point.pt.y - smax*sigma_size_) - 1;
            down_y = fRound(point.pt.y + smax*sigma_size_) + 1;

            if (left_x < 0 || right_x >= evolution_[i].Ldet.cols ||
                up_y < 0 || down_y >= evolution_[i].Ldet.rows) {
              is_out = true;
            }

            if (is_out == false) {
              if (is_repeated == false) {
                point.pt.x *= ratio;
                point.pt.y *= ratio;
                kpts_aux.push_back(point);
                npoints++;
              }
              else {
                point.pt.x *= ratio;
                point.pt.y *= ratio;
                kpts_aux[id_repeated] = point;
              }
            } // if is_out
          } //if is_extremum
        }
      } // for jx
      prev = curr;
      curr = next;
    } // for ix
  } // for i

  // Now filter points with the upper scale level
  for (size_t i = 0; i < kpts_aux.size(); i++) {

    is_repeated = false;
    const KeyPoint& pt = kpts_aux[i];
    for (size_t j = i + 1; j < kpts_aux.size(); j++) {

      // Compare response with the upper scale
      if ((pt.class_id + 1) == kpts_aux[j].class_id) {
        float distx = pt.pt.x - kpts_aux[j].pt.x;
        float disty = pt.pt.y - kpts_aux[j].pt.y;
        dist = distx * distx + disty * disty;
        if (dist <= pt.size * pt.size) {
          if (pt.response < kpts_aux[j].response) {
            is_repeated = true;
            break;
          }
        }
      }
    }

    if (is_repeated == false)
      kpts.push_back(pt);
  }
}

/* ************************************************************************* */
/**
 * @brief This method performs subpixel refinement of the detected keypoints
 * @param kpts Vector of detected keypoints
 */
void AKAZEFeatures::Do_Subpixel_Refinement(std::vector<KeyPoint>& kpts)
{
  float Dx = 0.0, Dy = 0.0, ratio = 0.0;
  float Dxx = 0.0, Dyy = 0.0, Dxy = 0.0;
  int x = 0, y = 0;
  Matx22f A(0, 0, 0, 0);
  Vec2f b(0, 0);
  Vec2f dst(0, 0);

  for (size_t i = 0; i < kpts.size(); i++) {
    ratio = (float)fastpow(2, kpts[i].octave);
    x = fRound(kpts[i].pt.x / ratio);
    y = fRound(kpts[i].pt.y / ratio);

    // Compute the gradient
    Dx = (0.5f)*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x + 1)
        - *(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x - 1));
    Dy = (0.5f)*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y + 1) + x)
        - *(evolution_[kpts[i].class_id].Ldet.ptr<float>(y - 1) + x));

    // Compute the Hessian
    Dxx = (*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x + 1)
        + *(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x - 1)
        - 2.0f*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x)));

    Dyy = (*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y + 1) + x)
        + *(evolution_[kpts[i].class_id].Ldet.ptr<float>(y - 1) + x)
        - 2.0f*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x)));

    Dxy = (0.25f)*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y + 1) + x + 1)
        + (*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y - 1) + x - 1)))
        - (0.25f)*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y - 1) + x + 1)
        + (*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y + 1) + x - 1)));

    // Solve the linear system
    A(0, 0) = Dxx;
    A(1, 1) = Dyy;
    A(0, 1) = A(1, 0) = Dxy;
    b(0) = -Dx;
    b(1) = -Dy;

    solve(A, b, dst, DECOMP_LU);

    if (fabs(dst(0)) <= 1.0f && fabs(dst(1)) <= 1.0f) {
        kpts[i].pt.x = x + dst(0);
      kpts[i].pt.y = y + dst(1);
      int power = fastpow(2, evolution_[kpts[i].class_id].octave);
      kpts[i].pt.x *= power;
      kpts[i].pt.y *= power;
      kpts[i].angle = 0.0;
	  // In OpenCV the size of a keypoint its the diameter
      kpts[i].size *= 2.0f;
    }
    // Delete the point since its not stable
    else {
      kpts.erase(kpts.begin() + i);
      i--;
    }
  }
}

/* ************************************************************************* */

class SURF_Descriptor_Upright_64_Invoker : public ParallelLoopBody
{
public:
  SURF_Descriptor_Upright_64_Invoker(std::vector<KeyPoint>& kpts, Mat& desc, std::vector<TEvolution>& evolution)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
  {
  }

  void operator() (const Range& range) const
  {
    for (int i = range.start; i < range.end; i++)
    {
      Get_SURF_Descriptor_Upright_64((*keypoints_)[i], descriptors_->ptr<float>(i));
    }
  }

  void Get_SURF_Descriptor_Upright_64(const KeyPoint& kpt, float* desc) const;

private:
  std::vector<KeyPoint>* keypoints_;
  Mat*                   descriptors_;
  std::vector<TEvolution>*   evolution_;
};

class SURF_Descriptor_64_Invoker : public ParallelLoopBody
{
public:
  SURF_Descriptor_64_Invoker(std::vector<KeyPoint>& kpts, Mat& desc, std::vector<TEvolution>& evolution)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
  {
  }

  void operator()(const Range& range) const
  {
    for (int i = range.start; i < range.end; i++)
    {
      AKAZEFeatures::Compute_Main_Orientation((*keypoints_)[i], *evolution_);
      Get_SURF_Descriptor_64((*keypoints_)[i], descriptors_->ptr<float>(i));
    }
  }

  void Get_SURF_Descriptor_64(const KeyPoint& kpt, float* desc) const;

private:
  std::vector<KeyPoint>* keypoints_;
  Mat*                   descriptors_;
  std::vector<TEvolution>*   evolution_;
};

class MSURF_Upright_Descriptor_64_Invoker : public ParallelLoopBody
{
public:
  MSURF_Upright_Descriptor_64_Invoker(std::vector<KeyPoint>& kpts, Mat& desc, std::vector<TEvolution>& evolution)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
  {
  }

  void operator()(const Range& range) const
  {
    for (int i = range.start; i < range.end; i++)
    {
      Get_MSURF_Upright_Descriptor_64((*keypoints_)[i], descriptors_->ptr<float>(i));
    }
  }

  void Get_MSURF_Upright_Descriptor_64(const KeyPoint& kpt, float* desc) const;

private:
  std::vector<KeyPoint>* keypoints_;
  Mat*                   descriptors_;
  std::vector<TEvolution>*   evolution_;
};

class MSURF_Descriptor_64_Invoker : public ParallelLoopBody
{
public:
  MSURF_Descriptor_64_Invoker(std::vector<KeyPoint>& kpts, Mat& desc, std::vector<TEvolution>& evolution)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
  {
  }

  void operator() (const Range& range) const
  {
    for (int i = range.start; i < range.end; i++)
    {
      AKAZEFeatures::Compute_Main_Orientation((*keypoints_)[i], *evolution_);
      Get_MSURF_Descriptor_64((*keypoints_)[i], descriptors_->ptr<float>(i));
    }
  }

  void Get_MSURF_Descriptor_64(const KeyPoint& kpt, float* desc) const;

private:
  std::vector<KeyPoint>* keypoints_;
  Mat*                   descriptors_;
  std::vector<TEvolution>*   evolution_;
};

class Upright_MLDB_Full_Descriptor_Invoker : public ParallelLoopBody
{
public:
  Upright_MLDB_Full_Descriptor_Invoker(std::vector<KeyPoint>& kpts, Mat& desc, std::vector<TEvolution>& evolution, AKAZEOptions& options)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
    , options_(&options)
  {
  }

  void operator() (const Range& range) const
  {
    for (int i = range.start; i < range.end; i++)
    {
      Get_Upright_MLDB_Full_Descriptor((*keypoints_)[i], descriptors_->ptr<unsigned char>(i));
    }
  }

  void Get_Upright_MLDB_Full_Descriptor(const KeyPoint& kpt, unsigned char* desc) const;

private:
  std::vector<KeyPoint>* keypoints_;
  Mat*                   descriptors_;
  std::vector<TEvolution>*   evolution_;
  AKAZEOptions*              options_;
};

class Upright_MLDB_Descriptor_Subset_Invoker : public ParallelLoopBody
{
public:
  Upright_MLDB_Descriptor_Subset_Invoker(std::vector<KeyPoint>& kpts,
                                         Mat& desc,
                                         std::vector<TEvolution>& evolution,
                                         AKAZEOptions& options,
                                         Mat descriptorSamples,
                                         Mat descriptorBits)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
    , options_(&options)
    , descriptorSamples_(descriptorSamples)
    , descriptorBits_(descriptorBits)
  {
  }

  void operator() (const Range& range) const
  {
    for (int i = range.start; i < range.end; i++)
    {
      Get_Upright_MLDB_Descriptor_Subset((*keypoints_)[i], descriptors_->ptr<unsigned char>(i));
    }
  }

  void Get_Upright_MLDB_Descriptor_Subset(const KeyPoint& kpt, unsigned char* desc) const;

private:
  std::vector<KeyPoint>* keypoints_;
  Mat*                   descriptors_;
  std::vector<TEvolution>*   evolution_;
  AKAZEOptions*              options_;

  Mat descriptorSamples_;  // List of positions in the grids to sample LDB bits from.
  Mat descriptorBits_;
};

class MLDB_Full_Descriptor_Invoker : public ParallelLoopBody
{
public:
  MLDB_Full_Descriptor_Invoker(std::vector<KeyPoint>& kpts, Mat& desc, std::vector<TEvolution>& evolution, AKAZEOptions& options)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
    , options_(&options)
  {
  }

  void operator() (const Range& range) const
  {
    for (int i = range.start; i < range.end; i++)
    {
      AKAZEFeatures::Compute_Main_Orientation((*keypoints_)[i], *evolution_);
      Get_MLDB_Full_Descriptor((*keypoints_)[i], descriptors_->ptr<unsigned char>(i));
    }
  }

  void Get_MLDB_Full_Descriptor(const KeyPoint& kpt, unsigned char* desc) const;
  void MLDB_Fill_Values(float* values, int sample_step, int level,
                        float xf, float yf, float co, float si, float scale) const;
  void MLDB_Binary_Comparisons(float* values, unsigned char* desc,
                               int count, int& dpos) const;

private:
  std::vector<KeyPoint>* keypoints_;
  Mat*                   descriptors_;
  std::vector<TEvolution>*   evolution_;
  AKAZEOptions*              options_;
};

class MLDB_Descriptor_Subset_Invoker : public ParallelLoopBody
{
public:
  MLDB_Descriptor_Subset_Invoker(std::vector<KeyPoint>& kpts,
                                 Mat& desc,
                                 std::vector<TEvolution>& evolution,
                                 AKAZEOptions& options,
                                 Mat descriptorSamples,
                                 Mat descriptorBits)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
    , options_(&options)
    , descriptorSamples_(descriptorSamples)
    , descriptorBits_(descriptorBits)
  {
  }

  void operator() (const Range& range) const
  {
    for (int i = range.start; i < range.end; i++)
    {
      AKAZEFeatures::Compute_Main_Orientation((*keypoints_)[i], *evolution_);
      Get_MLDB_Descriptor_Subset((*keypoints_)[i], descriptors_->ptr<unsigned char>(i));
    }
  }

  void Get_MLDB_Descriptor_Subset(const KeyPoint& kpt, unsigned char* desc) const;

private:
  std::vector<KeyPoint>* keypoints_;
  Mat*                   descriptors_;
  std::vector<TEvolution>*   evolution_;
  AKAZEOptions*              options_;

  Mat descriptorSamples_;  // List of positions in the grids to sample LDB bits from.
  Mat descriptorBits_;
};

/**
 * @brief This method  computes the set of descriptors through the nonlinear scale space
 * @param kpts Vector of detected keypoints
 * @param desc Matrix to store the descriptors
 */
void AKAZEFeatures::Compute_Descriptors(std::vector<KeyPoint>& kpts, Mat& desc)
{
  for(size_t i = 0; i < kpts.size(); i++)
  {
      CV_Assert(0 <= kpts[i].class_id && kpts[i].class_id < static_cast<int>(evolution_.size()));
  }

  // Allocate memory for the matrix with the descriptors
  if (options_.descriptor < AKAZE::DESCRIPTOR_MLDB_UPRIGHT) {
    desc = Mat::zeros((int)kpts.size(), 128, CV_32FC1); // nahum_dbg(2) 10/12/15
  }
  else {
    // We use the full length binary descriptor -> 486 bits
    if (options_.descriptor_size == 0) {
      int t = (6 + 36 + 120)*options_.descriptor_channels;
      desc = Mat::zeros((int)kpts.size(), (int)ceil(t / 8.), CV_8UC1);
    }
    else {
      // We use the random bit selection length binary descriptor
      desc = Mat::zeros((int)kpts.size(), (int)ceil(options_.descriptor_size / 8.), CV_8UC1);
    }
  }

  switch (options_.descriptor)
  {
    case AKAZE::DESCRIPTOR_KAZE_UPRIGHT: // Upright descriptors, not invariant to rotation
    {
      parallel_for_(Range(0, (int)kpts.size()), MSURF_Upright_Descriptor_64_Invoker(kpts, desc, evolution_));
    }
    break;
    case AKAZE::DESCRIPTOR_KAZE:
    {
     parallel_for_(Range(0, (int)kpts.size()), MSURF_Descriptor_64_Invoker(kpts, desc, evolution_));
    }
    break;
    case AKAZE::DESCRIPTOR_MLDB_UPRIGHT: // Upright descriptors, not invariant to rotation
    {
      if (options_.descriptor_size == 0)
        parallel_for_(Range(0, (int)kpts.size()), Upright_MLDB_Full_Descriptor_Invoker(kpts, desc, evolution_, options_));
      else
        parallel_for_(Range(0, (int)kpts.size()), Upright_MLDB_Descriptor_Subset_Invoker(kpts, desc, evolution_, options_, descriptorSamples_, descriptorBits_));
    }
    break;
    case AKAZE::DESCRIPTOR_MLDB:
    {
      if (options_.descriptor_size == 0)
        parallel_for_(Range(0, (int)kpts.size()), MLDB_Full_Descriptor_Invoker(kpts, desc, evolution_, options_));
      else
        parallel_for_(Range(0, (int)kpts.size()), MLDB_Descriptor_Subset_Invoker(kpts, desc, evolution_, options_, descriptorSamples_, descriptorBits_));
    }
    break;
  }
}

/* ************************************************************************* */
/**
 * @brief This method computes the main orientation for a given keypoint
 * @param kpt Input keypoint
 * @note The orientation is computed using a similar approach as described in the
 * original SURF method. See Bay et al., Speeded Up Robust Features, ECCV 2006
 */
float find_second_max(KeyPoint &kpt,float *mm,float *tt,int mmax,float ymax_first)
	 {
		
	 float x0,y0,x1,y1,x2,y2,a,b,c,xmax,ymax;
		  float max=0;
		   int idxMax=0;
		   int k=1;
		   int flag_in=0;
		  while(k<=43)
		  {
			  y0=mm[k-1];
			  y1=mm[k];
			  y2=mm[k+1];
			 
			  if((y0<y1)&&(y2<y1)&&(max<y1)&&(k!=mmax+1))
			  {
				 max=y1;
				idxMax=k-1;
				flag_in=1;
			  }
			  k++;
		  }
  if(flag_in==1)
	{
	 x0=tt[idxMax];
	 y0=mm[idxMax];
	 x1=tt[idxMax+1];
	 y1=mm[idxMax+1];
	 x2=tt[idxMax+2];
	 y2=mm[idxMax+2];

      a=((x1-x0)*(y2-y1)-(x2-x1)*(y1-y0))/((x2-x0)*(x1-x0)*(x2-x1));
	  if(abs(a)>0)
	  {
	  b=(y1-y0)/(x1-x0)-a*(x1+x0);
	  c=y1-a*x1*x1-b*x1;
	  xmax=-b/(2*a);
	  ymax=c-b*b/(4*a);
	  if(xmax>2*CV_PI)xmax=2*CV_PI-xmax;
	  if(xmax<0)xmax=2*CV_PI+xmax;
	  kpt.response=xmax;
	  }
	  else
	  {
		  ymax=0;
	  }
   }
  else
    {
     kpt.response=0;
     ymax=0;
    }
  if(ymax>ymax_first)
  {
	  float temp=kpt.angle;
	  kpt.angle=kpt.response;
	  kpt.response=temp;
  }
 
if((kpt.response>0)&&(ymax/ymax_first<0.95))
			kpt.response=0;
return ymax;

	 }
void find_min_max(KeyPoint &kpt,float *maxx,float *tetax,int idxMax,int max)
{
	float x0,y0,x1,y1,x2,y2,a,b,c,xmax,ymax;
	 int mmax=max;
	 
	 float *mm=new float[44];
	 mm[0]=maxx[41];
	 mm[43]=maxx[0];
	 float *tt=new float[44];
	tt[0]=tetax[41];
	tt[43]=tetax[0];
	 for(int i=0;i<42;i++)
		  {
			  mm[i+1]=maxx[i];
			  tt[i+1]=tetax[i];
			 
		  }
	  if(tt[0]>tt[1])tt[0]=tt[0]-2*CV_PI;
	  if(tt[43]<tt[42])tt[43]=2*CV_PI-tt[43];
	 x0=tt[idxMax];
	 y0=mm[idxMax];
	 x1=tt[idxMax+1];
	 y1=mm[idxMax+1];
	 x2=tt[idxMax+2];
	 y2=mm[idxMax+2];

      a=((x1-x0)*(y2-y1)-(x2-x1)*(y1-y0))/((x2-x0)*(x1-x0)*(x2-x1));
	  b=(y1-y0)/(x1-x0)-a*(x1+x0);
	  c=y1-a*x1*x1-b*x1;
	  xmax=-b/(2*a);
	  ymax=c-b*b/(4*a);
	  if(xmax>2*CV_PI)xmax=2*CV_PI-xmax;
	  if(xmax<0)xmax=2*CV_PI+xmax;
	 
	  if((abs(kpt.angle-xmax)<1.5)&&(ymax>0))
	  {
	     kpt.angle=xmax;
	  //  float ymax_second= find_second_max(kpt,mm,tt, idxMax,ymax);
		
		 
	/*	 int ratio=(int)((float)MIN(ymax,ymax_second)/((float)MAX(ymax,ymax_second))*65536+.5);
		 
	     if((abs(xmax-kpt.response)<0.15)&&(abs(ymax-ymax_second)/ymax<0.1))
		 {
			 float yavg=(ymax+ymax_second)/2;
			 a=(ymax-ymax_second)/(xmax-kpt.response);
			 b=ymax_second-kpt.response*(ymax-ymax_second)/(xmax-kpt.response);
			 float xavg=(yavg-b)/a;
		     kpt.angle=xavg;
			 kpt.response=0;
		 }
		  
		  if((kpt.response>0)&&(ratio>6536))
		  {
			  
			  if((kpt.angle-kpt.response)>CV_PI)
			  {
			      kpt.response=kpt.angle-kpt.response-CV_PI;
				
			  }
			 if((kpt.angle-kpt.response)<-CV_PI)
			      kpt.response=kpt.angle-kpt.response+CV_PI;

			   if(kpt.response>CV_PI)kpt.response=2*CV_PI-kpt.response;
			  kpt.octave=ratio;
			  int aa=1;
		  }
		  else
		  {
			  kpt.response=0;
            int aa=1;
		  }

	  }
	  else
	  {
		  kpt.response=0;
		  int aa=1;
		 */
	  }
	 delete []mm;
	 delete []tt;
	 
}

void AKAZEFeatures::Compute_Main_Orientation(KeyPoint& kpt, const std::vector<TEvolution>& evolution_)
{
    /* ************************************************************************* */
    /// Lookup table for 2d gaussian (sigma = 2.5) where (0,0) is top left and (6,6) is bottom right
	kpt.angle=kpt.angle*(3.14159265/180.0);
	if(kpt.response>0)
	{

    static const float gauss25[7][7] =
    {
        { 0.02546481f, 0.02350698f, 0.01849125f, 0.01239505f, 0.00708017f, 0.00344629f, 0.00142946f },
        { 0.02350698f, 0.02169968f, 0.01706957f, 0.01144208f, 0.00653582f, 0.00318132f, 0.00131956f },
        { 0.01849125f, 0.01706957f, 0.01342740f, 0.00900066f, 0.00514126f, 0.00250252f, 0.00103800f },
        { 0.01239505f, 0.01144208f, 0.00900066f, 0.00603332f, 0.00344629f, 0.00167749f, 0.00069579f },
        { 0.00708017f, 0.00653582f, 0.00514126f, 0.00344629f, 0.00196855f, 0.00095820f, 0.00039744f },
        { 0.00344629f, 0.00318132f, 0.00250252f, 0.00167749f, 0.00095820f, 0.00046640f, 0.00019346f },
        { 0.00142946f, 0.00131956f, 0.00103800f, 0.00069579f, 0.00039744f, 0.00019346f, 0.00008024f }
    };

  int ix = 0, iy = 0, idx = 0,  level = 0;
  float s=0;
  float xf = 0.0, yf = 0.0, gweight = 0.0, ratio = 0.0;
  const int ang_size = 109;
  float resX[ang_size], resY[ang_size], Ang[ang_size];
  const int id[] = { 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6 };
  float max1=0;
  float teta1=0;
  float teta=0;
  float rat=0;
  int idxMax=0;
 
  float maxx[42]={0};
  float tetax[42]={0};
 
  // Variables for computing the dominant direction
  float sumX = 0.0, sumY = 0.0, max = 0.0, ang1 = 0.0, ang2 = 0.0;
   level = kpt.class_id;
  float w=evolution_[level].Lx.cols;
  float h=evolution_[level].Lx.rows;
  // Get the information from the keypoint
 
  ratio = (float)(1 << evolution_[level].octave);
  s = (0.5f*kpt.size / ratio);
  xf = kpt.pt.x / ratio;
  yf = kpt.pt.y / ratio;

  // Calculate derivatives responses for points within radius of 6*scale
  for (int i = -6; i <= 6; ++i)
  {
    for (int j = -6; j <= 6; ++j) 
	{
      if (i*i + j*j < 36) 
	  {
        iy = MIN(MAX(fRound(yf + j*s),0),h-1);
        ix = MIN(MAX(fRound(xf + i*s),0),w-1);

      // gweight = gauss25[id[i + 6]][id[j + 6]];
		gweight=1.0;
        resX[idx] = gweight*(*(evolution_[level].Lx.ptr<float>(iy)+ix));
        resY[idx] = gweight*(*(evolution_[level].Ly.ptr<float>(iy)+ix));

        ++idx;
      }
    }
  }
  hal::fastAtan2(resY, resX, Ang, ang_size, false);
  // Loop slides pi/3 window around feature point
  idx=0;
  for (ang1 = 0; ang1 < (float)(2.0 * CV_PI); ang1 += 0.15)
  {
    ang2 = (ang1 + (float)(CV_PI / 3.0) >(float)(2.0*CV_PI) ? ang1 - (float)(5.0*CV_PI / 3.0) : ang1 + (float)(CV_PI / 3.0));
    sumX = sumY = 0.f;
	float sum=0;
    for (int k = 0; k < ang_size; ++k)
	{
      // Get angle from the x-axis of the sample point
      const float & ang = Ang[k];

      // Determine whether the point is within the window
      if (ang1 < ang2 && ang1 < ang && ang < ang2)
	  {
        sumX += resX[k];
        sumY += resY[k];
		
      }
      else if (ang2 < ang1 &&
               ((ang > 0 && ang < ang2) || (ang > ang1 && ang < 2.0f*CV_PI)))
	  {
        sumX += resX[k];
        sumY += resY[k];
		
      }
    }//for k
	maxx[idx]=sumX*sumX + sumY*sumY;
	tetax[idx]=getAngle(sumX, sumY);
	
	
    // if the vector produced from this window is longer than all
    // previous vectors then this forms the new dominant direction
    if (sumX*sumX + sumY*sumY > max)
   	{
      // store largest orientation
	 
	
	 idxMax=idx;
      max = sumX*sumX + sumY*sumY;
      kpt.angle = getAngle(sumX, sumY);
		    
    }
	idx++;
  }
 find_min_max(kpt,maxx,tetax, idxMax,max);
//  kpt.response=teta1;
//  kpt.octave=(int)(rat*65536+.5);
 
	
 
}//if kpts.response>0

}

/* ************************************************************************* */
/**
 * @brief This method computes the upright descriptor (not rotation invariant) of
 * the provided keypoint
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 * @note Rectangular grid of 24 s x 24 s. Descriptor Length 64. The descriptor is inspired
 * from Agrawal et al., CenSurE: Center Surround Extremas for Realtime Feature Detection and Matching,
 * ECCV 2008
 */
void MSURF_Upright_Descriptor_64_Invoker::Get_MSURF_Upright_Descriptor_64(const KeyPoint& kpt, float *desc) const {

  float dx = 0.0, dy = 0.0, mdx = 0.0, mdy = 0.0, gauss_s1 = 0.0, gauss_s2 = 0.0;
  float rx = 0.0, ry = 0.0, len = 0.0, xf = 0.0, yf = 0.0, ys = 0.0, xs = 0.0;
  float sample_x = 0.0, sample_y = 0.0;
  int x1 = 0, y1 = 0, sample_step = 0, pattern_size = 0;
  int x2 = 0, y2 = 0, kx = 0, ky = 0, i = 0, j = 0, dcount = 0;
  float fx = 0.0, fy = 0.0, ratio = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
  int scale = 0, dsize = 0, level = 0;

  // Subregion centers for the 4x4 gaussian weighting
  float cx = -0.5f, cy = 0.5f;

  const std::vector<TEvolution>& evolution = *evolution_;

  // Set the descriptor size and the sample and pattern sizes
  dsize = 64;
  sample_step = 5;
  pattern_size = 12;

  // Get the information from the keypoint
  ratio = (float)(1 << kpt.octave);
 
  scale = (0.5f*kpt.size / ratio);
  level = kpt.class_id;
  yf = kpt.pt.y / ratio;
  xf = kpt.pt.x / ratio;

  i = -8;

  // Calculate descriptor for this interest point
  // Area of size 24 s x 24 s
  while (i < pattern_size) {
    j = -8;
    i = i - 4;

    cx += 1.0f;
    cy = -0.5f;

    while (j < pattern_size) {
      dx = dy = mdx = mdy = 0.0;
      cy += 1.0f;
      j = j - 4;

      ky = i + sample_step;
      kx = j + sample_step;

      ys = yf + (ky*scale);
      xs = xf + (kx*scale);

      for (int k = i; k < i + 9; k++) {
        for (int l = j; l < j + 9; l++) {
          sample_y = k*scale + yf;
          sample_x = l*scale + xf;

          //Get the gaussian weighted x and y responses
          gauss_s1 = gaussian(xs - sample_x, ys - sample_y, 2.50f*scale);

          y1 = (int)(sample_y - .5);
          x1 = (int)(sample_x - .5);

          y2 = (int)(sample_y + .5);
          x2 = (int)(sample_x + .5);

          fx = sample_x - x1;
          fy = sample_y - y1;

          res1 = *(evolution[level].Lx.ptr<float>(y1)+x1);
          res2 = *(evolution[level].Lx.ptr<float>(y1)+x2);
          res3 = *(evolution[level].Lx.ptr<float>(y2)+x1);
          res4 = *(evolution[level].Lx.ptr<float>(y2)+x2);
          rx = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2 + (1.0f - fx)*fy*res3 + fx*fy*res4;

          res1 = *(evolution[level].Ly.ptr<float>(y1)+x1);
          res2 = *(evolution[level].Ly.ptr<float>(y1)+x2);
          res3 = *(evolution[level].Ly.ptr<float>(y2)+x1);
          res4 = *(evolution[level].Ly.ptr<float>(y2)+x2);
          ry = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2 + (1.0f - fx)*fy*res3 + fx*fy*res4;

          rx = gauss_s1*rx;
          ry = gauss_s1*ry;

          // Sum the derivatives to the cumulative descriptor
          dx += rx;
          dy += ry;
          mdx += fabs(rx);
          mdy += fabs(ry);
        }
      }

      // Add the values to the descriptor vector
      gauss_s2 = gaussian(cx - 2.0f, cy - 2.0f, 1.5f);

      desc[dcount++] = dx*gauss_s2;
      desc[dcount++] = dy*gauss_s2;
      desc[dcount++] = mdx*gauss_s2;
      desc[dcount++] = mdy*gauss_s2;

      len += (dx*dx + dy*dy + mdx*mdx + mdy*mdy)*gauss_s2*gauss_s2;

      j += 9;
    }

    i += 9;
  }

  // convert to unit vector
  len = sqrt(len);

  for (i = 0; i < dsize; i++) {
    desc[i] /= len;
  }
}

void MSURF_Descriptor_64_Invoker::Get_MSURF_Descriptor_64(const KeyPoint& kpt, float *desc) 
 const {
	/*static float m_data[128]=
                 {-0.00338813546 ,  -0.00349699124, -0.000447582192 , -0.000736448448  ,   0.0558114573  ,   0.0588222928   ,  0.0596071407   ,  0.0741576180,
	             -0.00170306512 ,  -0.00239867577 , -0.00354001159  , -0.00477006473  ,   0.0485182144   ,  0.0628645942   ,  0.0643333718   ,  0.0728200227,
				  0.00172934704 ,   0.00245559867 ,  -0.00458895974 , -0.00326277362  ,   0.0489259213  ,   0.0625307038   ,  0.0731955767   ,  0.0637607574 ,
				  0.00319408416  ,  0.00329247466 , -0.000636671495 , -0.000397112395  ,   0.0559423454  ,   0.0583358109  ,   0.0737030879   ,  0.0594402775,
				 -0.00475469930 ,  -0.00221227808 ,  0.0177943688   ,  0.0202471334  ,   0.0767624602  ,   0.0492728017  ,   0.0789484978   ,  0.0949267522, 
				 -0.00318503729 ,  -0.00176860031 ,  0.0410723016   ,  0.0445662253  ,   0.0713574216  ,   0.0461750478  ,    0.101950616   ,   0.110943951,
				  0.00356236962 ,   0.00179772254 ,  0.0413943492   ,  0.0388696454  ,   0.0726023763   ,  0.0460009351   ,   0.109013438    , 0.0984543115,
				  0.00446776533 ,   0.00203372119 ,  0.0193022173   ,  0.0168810748  ,   0.0758306235  ,   0.0492330641   ,  0.0933979824    , 0.0779747367 ,
				  0.00802469812 ,   0.00260450039 ,  0.0207315050   ,  0.0144363381  ,   0.0763439983  ,   0.0516602136   ,  0.0972985625   ,  0.0755338296,
				  0.00489012105 ,   0.00220368942 ,  0.0393115617   ,   0.0342139155  ,   0.0694338009  ,   0.0492064208    ,  0.109608091   ,  0.0970784202,
                 -0.00564122712 ,  -0.00222592661 ,   0.0322282538  ,   0.0370427705  ,   0.0707237646  ,   0.0491049923   ,  0.0934700295   ,   0.108702354 ,
				 -0.00757697783 ,  -0.00236307154 ,   0.0138388239  ,   0.0197107866  ,   0.0755087882  ,   0.0515678190   ,  0.0748301595    , 0.0955438167,
				  0.00414472353 ,   0.00401799707 ,  -0.000262248970 , -0.000122994810  ,   0.0574738570  ,   0.0605679117   ,  0.0766846463    , 0.0598674305,
				  0.00203560479 ,   0.00255784020 ,  -0.00388567569 ,  -0.00263729529  ,   0.0505272187  ,   0.0650754720   ,  0.0761063918    , 0.0662502870,
				 -0.00218184269 ,  -0.00270466111 ,  -0.00238398882  , -0.00379783520  ,   0.0509907901   ,  0.0647658631   ,  0.0654325262    , 0.0765611678,
				 -0.00396828074  , -0.00380070461 ,-4.43073732e-005 , -0.000203892385  ,   0.0576683022   ,  0.0599504635    , 0.0597355925    , 0.0761346370 };
	*/
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  float dxP = 0.0,dxM = 0.0, dyP = 0.0,dyM = 0.0, mdxP = 0.0, mdxM = 0.0,mdyP = 0.0, mdyM = 0.0,gauss_s1 = 0.0, gauss_s2 = 0.0;
   float rx = 0.0, ry = 0.0, rrx = 0.0, rry = 0.0, len = 0.0, xf = 0.0, yf = 0.0, ys = 0.0, xs = 0.0;
  float sample_x = 0.0, sample_y = 0.0, co = 0.0, si = 0.0, angle = 0.0;
  float fx = 0.0, fy = 0.0, ratio = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
  int x1 = 0, y1 = 0, x2 = 0, y2 = 0, sample_step = 0, pattern_size = 0;
  int kx = 0, ky = 0, i = 0, j = 0, dcount = 0;
  int  dsize = 0, level = 0;
  float scale;
  // Subregion centers for the 4x4 gaussian weighting
  float cx = -0.5f, cy = 0.5f;
 
  const std::vector<TEvolution>& evolution = *evolution_;

  // Set the descriptor size and the sample and pattern sizes
  dsize = 128;
//  if(abs(kpt.response>0.1))
//  {
  sample_step = 5;
  pattern_size = 12;

  // Get the information from the keypoint
  ratio = (float)(1 << kpt.octave);
 

 // ratio=(float) ( 1<<(kpt.class_id>>2));
  scale = (0.5f*kpt.size / ratio);
 
 
  level = kpt.class_id;
  float  w=evolution[level].Lx.cols;
  float  h=evolution[level].Lx.rows;
  yf = kpt.pt.y / ratio;
  xf = kpt.pt.x / ratio;
 // float del_angle=0.1;
 // int consistency_flag=1;
 // unsigned long index=0,index1=0;
//	unsigned char sign=0,sign1=0;
//	static float rr[3]={.9,1.0,1.1};
 //for (int ii=0;ii<3;ii++)
// {
   angle = kpt.angle;
   dcount=0;
 //  float scale1=scale*rr[ii];
  cx = -0.5f, cy = 0.5f;

  co = cos(angle);
  si = sin(angle);

  i = -8;

  // Calculate descriptor for this interest point
  // Area of size 24 s x 24 s
 
  while (i < pattern_size)
  {
    j = -8;
    i = i - 4;

    cx += 1.0f;
    cy = -0.5f;

    while (j < pattern_size)
	{
      dxP = dyP = mdxP = mdyP = 0.0;
	  dxM = dyM = mdxM = mdyM = 0.0;
	 
      cy += 1.0f;
      j = j - 4;

      ky = i + sample_step;
      kx = j + sample_step;

      xs = xf + (-kx*scale*si + ky*scale*co);
      ys = yf + (kx*scale*co + ky*scale*si);

      for (int k = i; k < i + 9; ++k)
	   {
        for (int l = j; l < j + 9; ++l) 
		{
          // Get coords of sample point on the rotated axis
          sample_y = yf + (l*scale*co + k*scale*si);
          sample_x = xf + (-l*scale*si + k*scale*co);

          // Get the gaussian weighted x and y responses
          gauss_s1 = gaussian(xs - sample_x, ys - sample_y, 2.5f*scale);
		 
          y1 = MIN(MAX(fRound(sample_y - 0.5f),0),h-1);
          x1 = MIN(MAX(fRound(sample_x - 0.5f),0),w-1);

          y2 = MIN(MAX(fRound(sample_y + 0.5f),0),h-1);
          x2 = MIN(MAX(fRound(sample_x + 0.5f),0),w-1);
		  

          fx = sample_x - x1;
          fy = sample_y - y1;

          res1 = *(evolution[level].Lx.ptr<float>(y1)+x1);
          res2 = *(evolution[level].Lx.ptr<float>(y1)+x2);
          res3 = *(evolution[level].Lx.ptr<float>(y2)+x1);
          res4 = *(evolution[level].Lx.ptr<float>(y2)+x2);
          rx = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2 + (1.0f - fx)*fy*res3 + fx*fy*res4;

          res1 = *(evolution[level].Ly.ptr<float>(y1)+x1);
          res2 = *(evolution[level].Ly.ptr<float>(y1)+x2);
          res3 = *(evolution[level].Ly.ptr<float>(y2)+x1);
          res4 = *(evolution[level].Ly.ptr<float>(y2)+x2);
          ry = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2 + (1.0f - fx)*fy*res3 + fx*fy*res4;

          // Get the x and y derivatives on the rotated axis
          rry = gauss_s1*(rx*co + ry*si);
          rrx = gauss_s1*(-rx*si + ry*co);

          // Sum the derivatives to the cumulative descriptor
		 // sum_dx+=rrx;
		//  sum_dy+=rry;
	  if(rry>=0)
		  {
           dxP += rrx;
		   mdxP += fabs(rrx);
		  }
		  else
		  {
            dxM += rrx;
		    mdxM += fabs(rrx);
		  }
		  
		 if(rrx>=0)
		  {
           dyP += rry;
		   mdyP += fabs(rry);
		  }
		  else
		  {
            dyM += rry;
		    mdyM += fabs(rry);
		  }	
          
        }
      }

      // Add the values to the descriptor vector
    // gauss_s2 = gaussian(cx - 2.0f, cy - 2.0f, 1.5f);

	
	  desc[dcount++] = dxP;
	  desc[dcount++] = dxM;
      desc[dcount++] = dyP;
	  desc[dcount++] = dyM;
      desc[dcount++] = mdxP;
	  desc[dcount++] = mdxM;
      desc[dcount++] = mdyP;
	  desc[dcount++] = mdyM;
	 
 //len += (dxP*dxP + dxM*dxM +dyP*dyP + dyM*dyM + mdxP*mdxP + mdxM*mdxM +mdyP*mdyP+mdyM*mdyM)*gauss_s2*gauss_s2;
   len += (dxP*dxP + dxM*dxM +dyP*dyP + dyM*dyM + mdxP*mdxP + mdxM*mdxM +mdyP*mdyP+mdyM*mdyM);

      j += 9;
    }

    i += 9;
  }

  // convert to unit vector
  len = sqrt(len);

    for (i = 0; i < dsize; i++)
     {
       desc[i] /= len;
     }
	///////////////////////////////////////////////////////////////////////////////
	/*
	 for (i = 0; i < dsize; i++)
     {
      sign=0;
	  index=find_index(desc,sign);
     }
	 
	 if(ii>0)
	 {
	 if((index1!=index)||(sign1!=sign))
		 consistency_flag=0;
	 }
	 index1=index;
	 sign1=sign;
	 }// ii loop
	 if(consistency_flag==1)
    int a=1;
	*/
 //////////////////////////////////////////////////////////////////////////////////
 }
/* ************************************************************************* */
/**
 * @brief This method computes the rupright descriptor (not rotation invariant) of
 * the provided keypoint
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 */
void Upright_MLDB_Full_Descriptor_Invoker::Get_Upright_MLDB_Full_Descriptor(const KeyPoint& kpt, unsigned char *desc) const {

  float di = 0.0, dx = 0.0, dy = 0.0;
  float ri = 0.0, rx = 0.0, ry = 0.0, xf = 0.0, yf = 0.0;
  float sample_x = 0.0, sample_y = 0.0, ratio = 0.0;
  int x1 = 0, y1 = 0, sample_step = 0, pattern_size = 0;
  int level = 0, nsamples = 0, scale = 0;
  int dcount1 = 0, dcount2 = 0;

  const AKAZEOptions & options = *options_;
  const std::vector<TEvolution>& evolution = *evolution_;

  // Matrices for the M-LDB descriptor
  Mat values_1 = Mat::zeros(4, options.descriptor_channels, CV_32FC1);
  Mat values_2 = Mat::zeros(9, options.descriptor_channels, CV_32FC1);
  Mat values_3 = Mat::zeros(16, options.descriptor_channels, CV_32FC1);

  // Get the information from the keypoint
  ratio = (float)(1 << kpt.octave);
  scale = fRound(0.5f*kpt.size / ratio);
  level = kpt.class_id;
  yf = kpt.pt.y / ratio;
  xf = kpt.pt.x / ratio;

  // First 2x2 grid
  pattern_size = options_->descriptor_pattern_size;
  sample_step = pattern_size;

  for (int i = -pattern_size; i < pattern_size; i += sample_step) {
    for (int j = -pattern_size; j < pattern_size; j += sample_step) {
      di = dx = dy = 0.0;
      nsamples = 0;

      for (int k = i; k < i + sample_step; k++) {
        for (int l = j; l < j + sample_step; l++) {

          // Get the coordinates of the sample point
          sample_y = yf + l*scale;
          sample_x = xf + k*scale;

          y1 = fRound(sample_y);
          x1 = fRound(sample_x);

          ri = *(evolution[level].Lt.ptr<float>(y1)+x1);
          rx = *(evolution[level].Lx.ptr<float>(y1)+x1);
          ry = *(evolution[level].Ly.ptr<float>(y1)+x1);

          di += ri;
          dx += rx;
          dy += ry;
          nsamples++;
        }
      }

      di /= nsamples;
      dx /= nsamples;
      dy /= nsamples;

      *(values_1.ptr<float>(dcount2)) = di;
      *(values_1.ptr<float>(dcount2)+1) = dx;
      *(values_1.ptr<float>(dcount2)+2) = dy;
      dcount2++;
    }
  }

  // Do binary comparison first level
  for (int i = 0; i < 4; i++) {
    for (int j = i + 1; j < 4; j++) {
      if (*(values_1.ptr<float>(i)) > *(values_1.ptr<float>(j))) {
        desc[dcount1 / 8] |= (1 << (dcount1 % 8));
      }
      dcount1++;

      if (*(values_1.ptr<float>(i)+1) > *(values_1.ptr<float>(j)+1)) {
        desc[dcount1 / 8] |= (1 << (dcount1 % 8));
      }
      dcount1++;

      if (*(values_1.ptr<float>(i)+2) > *(values_1.ptr<float>(j)+2)) {
        desc[dcount1 / 8] |= (1 << (dcount1 % 8));
      }
      dcount1++;
    }
  }

  // Second 3x3 grid
  sample_step = static_cast<int>(ceil(pattern_size*2. / 3.));
  dcount2 = 0;

  for (int i = -pattern_size; i < pattern_size; i += sample_step) {
    for (int j = -pattern_size; j < pattern_size; j += sample_step) {
      di = dx = dy = 0.0;
      nsamples = 0;

      for (int k = i; k < i + sample_step; k++) {
        for (int l = j; l < j + sample_step; l++) {

          // Get the coordinates of the sample point
          sample_y = yf + l*scale;
          sample_x = xf + k*scale;

          y1 = fRound(sample_y);
          x1 = fRound(sample_x);

          ri = *(evolution[level].Lt.ptr<float>(y1)+x1);
          rx = *(evolution[level].Lx.ptr<float>(y1)+x1);
          ry = *(evolution[level].Ly.ptr<float>(y1)+x1);

          di += ri;
          dx += rx;
          dy += ry;
          nsamples++;
        }
      }

      di /= nsamples;
      dx /= nsamples;
      dy /= nsamples;

      *(values_2.ptr<float>(dcount2)) = di;
      *(values_2.ptr<float>(dcount2)+1) = dx;
      *(values_2.ptr<float>(dcount2)+2) = dy;
      dcount2++;
    }
  }

  //Do binary comparison second level
  dcount2 = 0;
  for (int i = 0; i < 9; i++) {
    for (int j = i + 1; j < 9; j++) {
      if (*(values_2.ptr<float>(i)) > *(values_2.ptr<float>(j))) {
        desc[dcount1 / 8] |= (1 << (dcount1 % 8));
      }
      dcount1++;

      if (*(values_2.ptr<float>(i)+1) > *(values_2.ptr<float>(j)+1)) {
        desc[dcount1 / 8] |= (1 << (dcount1 % 8));
      }
      dcount1++;

      if (*(values_2.ptr<float>(i)+2) > *(values_2.ptr<float>(j)+2)) {
        desc[dcount1 / 8] |= (1 << (dcount1 % 8));
      }
      dcount1++;
    }
  }

  // Third 4x4 grid
  sample_step = pattern_size / 2;
  dcount2 = 0;

  for (int i = -pattern_size; i < pattern_size; i += sample_step) {
    for (int j = -pattern_size; j < pattern_size; j += sample_step) {
      di = dx = dy = 0.0;
      nsamples = 0;

      for (int k = i; k < i + sample_step; k++) {
        for (int l = j; l < j + sample_step; l++) {

          // Get the coordinates of the sample point
          sample_y = yf + l*scale;
          sample_x = xf + k*scale;

          y1 = fRound(sample_y);
          x1 = fRound(sample_x);

          ri = *(evolution[level].Lt.ptr<float>(y1)+x1);
          rx = *(evolution[level].Lx.ptr<float>(y1)+x1);
          ry = *(evolution[level].Ly.ptr<float>(y1)+x1);

          di += ri;
          dx += rx;
          dy += ry;
          nsamples++;
        }
      }

      di /= nsamples;
      dx /= nsamples;
      dy /= nsamples;

      *(values_3.ptr<float>(dcount2)) = di;
      *(values_3.ptr<float>(dcount2)+1) = dx;
      *(values_3.ptr<float>(dcount2)+2) = dy;
      dcount2++;
    }
  }

  //Do binary comparison third level
  dcount2 = 0;
  for (int i = 0; i < 16; i++) {
    for (int j = i + 1; j < 16; j++) {
      if (*(values_3.ptr<float>(i)) > *(values_3.ptr<float>(j))) {
        desc[dcount1 / 8] |= (1 << (dcount1 % 8));
      }
      dcount1++;

      if (*(values_3.ptr<float>(i)+1) > *(values_3.ptr<float>(j)+1)) {
        desc[dcount1 / 8] |= (1 << (dcount1 % 8));
      }
      dcount1++;

      if (*(values_3.ptr<float>(i)+2) > *(values_3.ptr<float>(j)+2)) {
        desc[dcount1 / 8] |= (1 << (dcount1 % 8));
      }
      dcount1++;
    }
  }
}

void MLDB_Full_Descriptor_Invoker::MLDB_Fill_Values(float* values, int sample_step, int level,
                                                    float xf, float yf, float co, float si, float scale) const
{
    const std::vector<TEvolution>& evolution = *evolution_;
    int pattern_size = options_->descriptor_pattern_size;
    int chan = options_->descriptor_channels;
    int valpos = 0;

    for (int i = -pattern_size; i < pattern_size; i += sample_step) {
        for (int j = -pattern_size; j < pattern_size; j += sample_step) {
            float di, dx, dy;
            di = dx = dy = 0.0;
            int nsamples = 0;

            for (int k = i; k < i + sample_step; k++) {
              for (int l = j; l < j + sample_step; l++) {
                float sample_y = yf + (l*co * scale + k*si*scale);
                float sample_x = xf + (-l*si * scale + k*co*scale);

                int y1 = fRound(sample_y);
                int x1 = fRound(sample_x);

                float ri = *(evolution[level].Lt.ptr<float>(y1)+x1);
                di += ri;

                if(chan > 1) {
                    float rx = *(evolution[level].Lx.ptr<float>(y1)+x1);
                    float ry = *(evolution[level].Ly.ptr<float>(y1)+x1);
                    if (chan == 2) {
                      dx += sqrtf(rx*rx + ry*ry);
                    }
                    else {
                      float rry = rx*co + ry*si;
                      float rrx = -rx*si + ry*co;
                      dx += rrx;
                      dy += rry;
                    }
                }
                nsamples++;
              }
            }
            di /= nsamples;
            dx /= nsamples;
            dy /= nsamples;

            values[valpos] = di;
            if (chan > 1) {
                values[valpos + 1] = dx;
            }
            if (chan > 2) {
              values[valpos + 2] = dy;
            }
            valpos += chan;
          }
        }
}

void MLDB_Full_Descriptor_Invoker::MLDB_Binary_Comparisons(float* values, unsigned char* desc,
                                                           int count, int& dpos) const {
    int chan = options_->descriptor_channels;
    int* ivalues = (int*) values;
    for(int i = 0; i < count * chan; i++) {
        ivalues[i] = CV_TOGGLE_FLT(ivalues[i]);
    }

    for(int pos = 0; pos < chan; pos++) {
        for (int i = 0; i < count; i++) {
            int ival = ivalues[chan * i + pos];
            for (int j = i + 1; j < count; j++) {
                int res = ival > ivalues[chan * j + pos];
                desc[dpos >> 3] |= (res << (dpos & 7));
                dpos++;
            }
        }
    }
}

/* ************************************************************************* */
/**
 * @brief This method computes the descriptor of the provided keypoint given the
 * main orientation of the keypoint
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 */
void MLDB_Full_Descriptor_Invoker::Get_MLDB_Full_Descriptor(const KeyPoint& kpt, unsigned char *desc) const {

  const int max_channels = 3;
  CV_Assert(options_->descriptor_channels <= max_channels);
  float values[16*max_channels];
  const double size_mult[3] = {1, 2.0/3.0, 1.0/2.0};

  float ratio = (float)(1 << kpt.octave);
  float scale = (float)fRound(0.5f*kpt.size / ratio);
  float xf = kpt.pt.x / ratio;
  float yf = kpt.pt.y / ratio;
  float co = cos(kpt.angle);
  float si = sin(kpt.angle);
  int pattern_size = options_->descriptor_pattern_size;

  int dpos = 0;
  for(int lvl = 0; lvl < 3; lvl++) {

      int val_count = (lvl + 2) * (lvl + 2);
      int sample_step = static_cast<int>(ceil(pattern_size * size_mult[lvl]));
      MLDB_Fill_Values(values, sample_step, kpt.class_id, xf, yf, co, si, scale);
      MLDB_Binary_Comparisons(values, desc, val_count, dpos);
  }
}

/* ************************************************************************* */
/**
 * @brief This method computes the M-LDB descriptor of the provided keypoint given the
 * main orientation of the keypoint. The descriptor is computed based on a subset of
 * the bits of the whole descriptor
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 */
void MLDB_Descriptor_Subset_Invoker::Get_MLDB_Descriptor_Subset(const KeyPoint& kpt, unsigned char *desc) const {

  float di = 0.f, dx = 0.f, dy = 0.f;
  float rx = 0.f, ry = 0.f;
  float sample_x = 0.f, sample_y = 0.f;
  int x1 = 0, y1 = 0;

  const AKAZEOptions & options = *options_;
  const std::vector<TEvolution>& evolution = *evolution_;

  // Get the information from the keypoint
  float ratio = (float)(1 << kpt.octave);
  int scale = fRound(0.5f*kpt.size / ratio);
  float angle = kpt.angle;
  int level = kpt.class_id;
  float yf = kpt.pt.y / ratio;
  float xf = kpt.pt.x / ratio;
  float co = cos(angle);
  float si = sin(angle);

  // Allocate memory for the matrix of values
  Mat values = Mat_<float>::zeros((4 + 9 + 16)*options.descriptor_channels, 1);

  // Sample everything, but only do the comparisons
  vector<int> steps(3);
  steps.at(0) = options.descriptor_pattern_size;
  steps.at(1) = (int)ceil(2.f*options.descriptor_pattern_size / 3.f);
  steps.at(2) = options.descriptor_pattern_size / 2;

  for (int i = 0; i < descriptorSamples_.rows; i++) {
    const int *coords = descriptorSamples_.ptr<int>(i);
    int sample_step = steps.at(coords[0]);
    di = 0.0f;
    dx = 0.0f;
    dy = 0.0f;

    for (int k = coords[1]; k < coords[1] + sample_step; k++) {
      for (int l = coords[2]; l < coords[2] + sample_step; l++) {

        // Get the coordinates of the sample point
        sample_y = yf + (l*scale*co + k*scale*si);
        sample_x = xf + (-l*scale*si + k*scale*co);

        y1 = fRound(sample_y);
        x1 = fRound(sample_x);

        di += *(evolution[level].Lt.ptr<float>(y1)+x1);

        if (options.descriptor_channels > 1) {
          rx = *(evolution[level].Lx.ptr<float>(y1)+x1);
          ry = *(evolution[level].Ly.ptr<float>(y1)+x1);

          if (options.descriptor_channels == 2) {
            dx += sqrtf(rx*rx + ry*ry);
          }
          else if (options.descriptor_channels == 3) {
            // Get the x and y derivatives on the rotated axis
            dx += rx*co + ry*si;
            dy += -rx*si + ry*co;
          }
        }
      }
    }

    *(values.ptr<float>(options.descriptor_channels*i)) = di;

    if (options.descriptor_channels == 2) {
      *(values.ptr<float>(options.descriptor_channels*i + 1)) = dx;
    }
    else if (options.descriptor_channels == 3) {
      *(values.ptr<float>(options.descriptor_channels*i + 1)) = dx;
      *(values.ptr<float>(options.descriptor_channels*i + 2)) = dy;
    }
  }

  // Do the comparisons
  const float *vals = values.ptr<float>(0);
  const int *comps = descriptorBits_.ptr<int>(0);

  for (int i = 0; i<descriptorBits_.rows; i++) {
    if (vals[comps[2 * i]] > vals[comps[2 * i + 1]]) {
      desc[i / 8] |= (1 << (i % 8));
    }
  }
}

/* ************************************************************************* */
/**
 * @brief This method computes the upright (not rotation invariant) M-LDB descriptor
 * of the provided keypoint given the main orientation of the keypoint.
 * The descriptor is computed based on a subset of the bits of the whole descriptor
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 */
void Upright_MLDB_Descriptor_Subset_Invoker::Get_Upright_MLDB_Descriptor_Subset(const KeyPoint& kpt, unsigned char *desc) const {

  float di = 0.0f, dx = 0.0f, dy = 0.0f;
  float rx = 0.0f, ry = 0.0f;
  float sample_x = 0.0f, sample_y = 0.0f;
  int x1 = 0, y1 = 0;

  const AKAZEOptions & options = *options_;
  const std::vector<TEvolution>& evolution = *evolution_;

  // Get the information from the keypoint
  float ratio = (float)(1 << kpt.octave);
  int scale = fRound(0.5f*kpt.size / ratio);
  int level = kpt.class_id;
  float yf = kpt.pt.y / ratio;
  float xf = kpt.pt.x / ratio;

  // Allocate memory for the matrix of values
  Mat values = Mat_<float>::zeros((4 + 9 + 16)*options.descriptor_channels, 1);

  vector<int> steps(3);
  steps.at(0) = options.descriptor_pattern_size;
  steps.at(1) = static_cast<int>(ceil(2.f*options.descriptor_pattern_size / 3.f));
  steps.at(2) = options.descriptor_pattern_size / 2;

  for (int i = 0; i < descriptorSamples_.rows; i++) {
    const int *coords = descriptorSamples_.ptr<int>(i);
    int sample_step = steps.at(coords[0]);
    di = 0.0f, dx = 0.0f, dy = 0.0f;

    for (int k = coords[1]; k < coords[1] + sample_step; k++) {
      for (int l = coords[2]; l < coords[2] + sample_step; l++) {

        // Get the coordinates of the sample point
        sample_y = yf + l*scale;
        sample_x = xf + k*scale;

        y1 = fRound(sample_y);
        x1 = fRound(sample_x);
        di += *(evolution[level].Lt.ptr<float>(y1)+x1);

        if (options.descriptor_channels > 1) {
          rx = *(evolution[level].Lx.ptr<float>(y1)+x1);
          ry = *(evolution[level].Ly.ptr<float>(y1)+x1);

          if (options.descriptor_channels == 2) {
            dx += sqrtf(rx*rx + ry*ry);
          }
          else if (options.descriptor_channels == 3) {
            dx += rx;
            dy += ry;
          }
        }
      }
    }

    *(values.ptr<float>(options.descriptor_channels*i)) = di;

    if (options.descriptor_channels == 2) {
      *(values.ptr<float>(options.descriptor_channels*i + 1)) = dx;
    }
    else if (options.descriptor_channels == 3) {
      *(values.ptr<float>(options.descriptor_channels*i + 1)) = dx;
      *(values.ptr<float>(options.descriptor_channels*i + 2)) = dy;
    }
  }

  // Do the comparisons
  const float *vals = values.ptr<float>(0);
  const int *comps = descriptorBits_.ptr<int>(0);

  for (int i = 0; i<descriptorBits_.rows; i++) {
    if (vals[comps[2 * i]] > vals[comps[2 * i + 1]]) {
      desc[i / 8] |= (1 << (i % 8));
    }
  }
}

/* ************************************************************************* */
/**
 * @brief This function computes a (quasi-random) list of bits to be taken
 * from the full descriptor. To speed the extraction, the function creates
 * a list of the samples that are involved in generating at least a bit (sampleList)
 * and a list of the comparisons between those samples (comparisons)
 * @param sampleList
 * @param comparisons The matrix with the binary comparisons
 * @param nbits The number of bits of the descriptor
 * @param pattern_size The pattern size for the binary descriptor
 * @param nchannels Number of channels to consider in the descriptor (1-3)
 * @note The function keeps the 18 bits (3-channels by 6 comparisons) of the
 * coarser grid, since it provides the most robust estimations
 */
void generateDescriptorSubsample(Mat& sampleList, Mat& comparisons, int nbits,
                                 int pattern_size, int nchannels) {

  int ssz = 0;
  for (int i = 0; i < 3; i++) {
    int gz = (i + 2)*(i + 2);
    ssz += gz*(gz - 1) / 2;
  }
  ssz *= nchannels;

  CV_Assert(nbits <= ssz); // Descriptor size can't be bigger than full descriptor

  // Since the full descriptor is usually under 10k elements, we pick
  // the selection from the full matrix.  We take as many samples per
  // pick as the number of channels. For every pick, we
  // take the two samples involved and put them in the sampling list

  Mat_<int> fullM(ssz / nchannels, 5);
  for (int i = 0, c = 0; i < 3; i++) {
    int gdiv = i + 2; //grid divisions, per row
    int gsz = gdiv*gdiv;
    int psz = (int)ceil(2.f*pattern_size / (float)gdiv);

    for (int j = 0; j < gsz; j++) {
      for (int k = j + 1; k < gsz; k++, c++) {
        fullM(c, 0) = i;
        fullM(c, 1) = psz*(j % gdiv) - pattern_size;
        fullM(c, 2) = psz*(j / gdiv) - pattern_size;
        fullM(c, 3) = psz*(k % gdiv) - pattern_size;
        fullM(c, 4) = psz*(k / gdiv) - pattern_size;
      }
    }
  }

  srand(1024);
  Mat_<int> comps = Mat_<int>(nchannels * (int)ceil(nbits / (float)nchannels), 2);
  comps = 1000;

  // Select some samples. A sample includes all channels
  int count = 0;
  int npicks = (int)ceil(nbits / (float)nchannels);
  Mat_<int> samples(29, 3);
  Mat_<int> fullcopy = fullM.clone();
  samples = -1;

  for (int i = 0; i < npicks; i++) {
    int k = rand() % (fullM.rows - i);
    if (i < 6) {
      // Force use of the coarser grid values and comparisons
      k = i;
    }

    bool n = true;

    for (int j = 0; j < count; j++) {
      if (samples(j, 0) == fullcopy(k, 0) && samples(j, 1) == fullcopy(k, 1) && samples(j, 2) == fullcopy(k, 2)) {
        n = false;
        comps(i*nchannels, 0) = nchannels*j;
        comps(i*nchannels + 1, 0) = nchannels*j + 1;
        comps(i*nchannels + 2, 0) = nchannels*j + 2;
        break;
      }
    }

    if (n) {
      samples(count, 0) = fullcopy(k, 0);
      samples(count, 1) = fullcopy(k, 1);
      samples(count, 2) = fullcopy(k, 2);
      comps(i*nchannels, 0) = nchannels*count;
      comps(i*nchannels + 1, 0) = nchannels*count + 1;
      comps(i*nchannels + 2, 0) = nchannels*count + 2;
      count++;
    }

    n = true;
    for (int j = 0; j < count; j++) {
      if (samples(j, 0) == fullcopy(k, 0) && samples(j, 1) == fullcopy(k, 3) && samples(j, 2) == fullcopy(k, 4)) {
        n = false;
        comps(i*nchannels, 1) = nchannels*j;
        comps(i*nchannels + 1, 1) = nchannels*j + 1;
        comps(i*nchannels + 2, 1) = nchannels*j + 2;
        break;
      }
    }

    if (n) {
      samples(count, 0) = fullcopy(k, 0);
      samples(count, 1) = fullcopy(k, 3);
      samples(count, 2) = fullcopy(k, 4);
      comps(i*nchannels, 1) = nchannels*count;
      comps(i*nchannels + 1, 1) = nchannels*count + 1;
      comps(i*nchannels + 2, 1) = nchannels*count + 2;
      count++;
    }

    Mat tmp = fullcopy.row(k);
    fullcopy.row(fullcopy.rows - i - 1).copyTo(tmp);
  }

  sampleList = samples.rowRange(0, count).clone();
  comparisons = comps.rowRange(0, nbits).clone();
}

}
