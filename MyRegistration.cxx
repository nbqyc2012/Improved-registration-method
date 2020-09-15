#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <vector>
#include "itksys/SystemTools.hxx"
#include<windows.h>
#include<fstream>
using namespace std;
#include "itkAffineTransform.h"
#include "itkCenteredTransformInitializer.h"
#include "itkMultiResolutionImageRegistrationMethod.h"
#include "NewMetric.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkRecursiveMultiResolutionPyramidImageFilter.h"
#include "itkImage.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkCheckerBoardImageFilter.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkCommand.h"

class CommandIterationUpdate : public itk::Command
{
public:
  typedef  CommandIterationUpdate   Self;
  typedef  itk::Command             Superclass;
  typedef  itk::SmartPointer<Self>  Pointer;
  itkNewMacro( Self );

protected:
  CommandIterationUpdate(): m_CumulativeIterationIndex(0) {};

public:
  typedef   itk::RegularStepGradientDescentOptimizer  OptimizerType;
  typedef   const OptimizerType *                     OptimizerPointer;

  void Execute(itk::Object *caller, const itk::EventObject & event) ITK_OVERRIDE
    {
    Execute( (const itk::Object *)caller, event);
    }

  void Execute(const itk::Object * object, const itk::EventObject & event) ITK_OVERRIDE
    {
    OptimizerPointer optimizer = static_cast< OptimizerPointer >( object );
    if( !(itk::IterationEvent().CheckEvent( &event )) )
      {
      return;
      }
    std::cout << optimizer->GetCurrentIteration() << "   ";
    std::cout << optimizer->GetValue() << "   ";
    std::cout << optimizer->GetCurrentPosition() << "  " <<m_CumulativeIterationIndex++ ;

	vnl_matrix<double> p(2, 2);
	p[0][0] = (double)optimizer->GetCurrentPosition()[0];
	p[0][1] = (double)optimizer->GetCurrentPosition()[1];
	p[1][0] = (double)optimizer->GetCurrentPosition()[2];
	p[1][1] = (double)optimizer->GetCurrentPosition()[3];
	vnl_svd<double> svd(p);
	vnl_matrix<double> r(2, 2);
	r = svd.U() * vnl_transpose(svd.V());
	double angle = std::asin(r[1][0]);
	std::cout << "    AffineAngle: " << angle * 180.0 / itk::Math::pi << std::endl;
    }

private:
  unsigned int m_CumulativeIterationIndex;
};

template <typename TRegistration>
class RegistrationInterfaceCommand : public itk::Command
{
public:
  typedef  RegistrationInterfaceCommand   Self;
  typedef  itk::Command                   Superclass;
  typedef  itk::SmartPointer<Self>        Pointer;
  itkNewMacro( Self );

protected:
  RegistrationInterfaceCommand() {};

public:
  typedef   TRegistration                              RegistrationType;
  typedef   RegistrationType *                         RegistrationPointer;
  typedef   itk::RegularStepGradientDescentOptimizer   OptimizerType;
  typedef   OptimizerType *                            OptimizerPointer;
  void Execute(itk::Object * object, const itk::EventObject & event) ITK_OVERRIDE
  {
    if( !(itk::IterationEvent().CheckEvent( &event )) )
      {
      return;
      }
    RegistrationPointer registration = static_cast<RegistrationPointer>( object );
    OptimizerPointer optimizer =
      static_cast< OptimizerPointer >( registration->GetModifiableOptimizer() );

    std::cout << "-------------------------------------" << std::endl;
    std::cout << "MultiResolution Level : "
              << registration->GetCurrentLevel()  << std::endl;
    std::cout << std::endl;

    if ( registration->GetCurrentLevel() == 0 )
      {
      optimizer->SetMaximumStepLength( 16.00 );
      optimizer->SetMinimumStepLength(  0.01 );
      }
    else
      {
      optimizer->SetMaximumStepLength( optimizer->GetMaximumStepLength() / 4.0 );
      optimizer->SetMinimumStepLength( optimizer->GetMinimumStepLength() / 10.0 );
      }
  }
  void Execute(const itk::Object * , const itk::EventObject & ) ITK_OVERRIDE
    { return; }
};

int main( int argc, char *argv[] )
{
   if( argc < 6 )
	   {
	   std::cerr << "Missing Parameters " << std::endl;
	   std::cerr << "Usage: " << argv[0];
	   std::cerr << " fixedImageFile  movingImageFile ";
	   std::cerr << " outputImagefile [backgroundGrayLevel]";
	   std::cerr << " [checkerboardbefore] [CheckerBoardAfter]";
	   return EXIT_FAILURE;
	   }
  long t1 = GetTickCount();//程序段开始前取得系统运行时间(ms) 
  const    unsigned int    Dimension = 2;
  typedef  unsigned short  PixelType;

  typedef itk::Image< PixelType, Dimension >  FixedImageType;
  typedef itk::Image< PixelType, Dimension >  MovingImageType;

  typedef   float                                    InternalPixelType;
  typedef itk::Image< InternalPixelType, Dimension > InternalImageType;
  typedef itk::AffineTransform< double, Dimension > TransformType;

  typedef itk::RegularStepGradientDescentOptimizer       OptimizerType;
  typedef itk::LinearInterpolateImageFunction<
                                    InternalImageType,
                                    double             > InterpolatorType;
  typedef itk::MattesMutualInformationImageToImageMetric<
                                          InternalImageType,
                                          InternalImageType >    MetricType;
  /*typedef itk::MeanSquaresImageToImageMetric<
	  InternalImageType,
	  InternalImageType >    MetricType;*/

  typedef OptimizerType::ScalesType       OptimizerScalesType;

  typedef itk::MultiResolutionImageRegistrationMethod<
                                    InternalImageType,
                                    InternalImageType    > RegistrationType;

  OptimizerType::Pointer      optimizer     = OptimizerType::New();
  InterpolatorType::Pointer   interpolator  = InterpolatorType::New();
  RegistrationType::Pointer   registration  = RegistrationType::New();
  MetricType::Pointer         metric        = MetricType::New();

  registration->SetOptimizer(     optimizer     );
  registration->SetInterpolator(  interpolator  );
  registration->SetMetric( metric  );

  TransformType::Pointer   transform  = TransformType::New();
  registration->SetTransform( transform );

  typedef itk::ImageFileReader< FixedImageType  > FixedImageReaderType;
  typedef itk::ImageFileReader< MovingImageType > MovingImageReaderType;

  FixedImageReaderType::Pointer  fixedImageReader  = FixedImageReaderType::New();
  MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();

  fixedImageReader->SetFileName(argv[1]);
  movingImageReader->SetFileName(argv[2]);

  typedef itk::CastImageFilter<
                        FixedImageType, InternalImageType > FixedCastFilterType;
  typedef itk::CastImageFilter<
                        MovingImageType, InternalImageType > MovingCastFilterType;
  FixedCastFilterType::Pointer fixedCaster   = FixedCastFilterType::New();
  MovingCastFilterType::Pointer movingCaster = MovingCastFilterType::New();

  fixedCaster->SetInput(  fixedImageReader->GetOutput() );
  movingCaster->SetInput( movingImageReader->GetOutput() );

  registration->SetFixedImage(    fixedCaster->GetOutput()    );
  registration->SetMovingImage(   movingCaster->GetOutput()   );

  fixedCaster->Update();

  registration->SetFixedImageRegion(
       fixedCaster->GetOutput()->GetBufferedRegion() );

  typedef itk::CenteredTransformInitializer<
            TransformType, 
			FixedImageType,
            MovingImageType >  TransformInitializerType;
  TransformInitializerType::Pointer initializer = TransformInitializerType::New();
  initializer->SetTransform(   transform );
  initializer->SetFixedImage(  fixedImageReader->GetOutput() );
  initializer->SetMovingImage( movingImageReader->GetOutput() );
  initializer->GeometryOn(); 
  initializer->InitializeTransform();
  registration->SetInitialTransformParameters( transform->GetParameters() );
  OptimizerScalesType optimizerScales( transform->GetNumberOfParameters() );

  optimizerScales[0] = 1.0; 
  optimizerScales[1] = 1.0; 
  optimizerScales[2] = 1.0; 
  optimizerScales[3] = 1.0; 
  optimizerScales[4] = 1.0 / 50000; 
  optimizerScales[5] = 1.0 / 50000; 

  optimizer->SetScales( optimizerScales );
  metric->SetNumberOfSpatialSamples( 50000 );
  metric->ReinitializeSeed( 76926294 );
  optimizer->SetNumberOfIterations(  2000  );
  optimizer->SetRelaxationFactor( 0.8 );
  optimizer->MinimizeOn();
  CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
  optimizer->AddObserver( itk::IterationEvent(), observer );

  typedef RegistrationInterfaceCommand<RegistrationType> CommandType;
  CommandType::Pointer command = CommandType::New();
  registration->AddObserver( itk::IterationEvent(), command );
  registration->SetNumberOfLevels( 3 );
  registration->Update();

  std::cout << "Optimizer Stopping Condition = " << optimizer->GetStopCondition() << std::endl;

  typedef RegistrationType::ParametersType ParametersType;
  ParametersType finalParameters = registration->GetLastTransformParameters();
    const double finalRotationCenterX = transform->GetCenter()[0];
  const double finalRotationCenterY = transform->GetCenter()[1];
  double TranslationAlongX = finalParameters[4];
  double TranslationAlongY = finalParameters[5];

  unsigned int numberOfIterations = optimizer->GetCurrentIteration();
  double bestValue = optimizer->GetValue();

  std::cout << "Result = " << std::endl;
  std::cout << " Center X      = " << finalRotationCenterX << std::endl;
  std::cout << " Center Y      = " << finalRotationCenterY << std::endl;
  std::cout << " Translation X = " << TranslationAlongX  << std::endl;
  std::cout << " Translation Y = " << TranslationAlongY  << std::endl;
  std::cout << " Iterations    = " << numberOfIterations << std::endl;
  std::cout << " Metric value  = " << bestValue          << std::endl;

  vnl_matrix<double> p(2, 2);
  p[0][0] = (double)finalParameters[0];
  p[0][1] = (double)finalParameters[1];
  p[1][0] = (double)finalParameters[2];
  p[1][1] = (double)finalParameters[3];
  vnl_svd<double> svd(p);
  vnl_matrix<double> r(2, 2);
  r = svd.U() * vnl_transpose(svd.V());
  double angle = std::asin(r[1][0]);
  const double angleInDegrees = angle * 180.0 / itk::Math::pi;

  std::cout << " Scale 1         = " << svd.W(0) << std::endl;
  std::cout << " Scale 2         = " << svd.W(1) << std::endl;
  std::cout << " Angle (degrees) = " << angleInDegrees << std::endl;
  TransformType::MatrixType matrix = transform->GetMatrix();
  TransformType::OffsetType offset = transform->GetOffset();
  std::cout << "Matrix = " << std::endl << matrix << std::endl;
  std::cout << "Offset = " << std::endl << offset << std::endl;
  
  typedef itk::ResampleImageFilter<MovingImageType, FixedImageType > ResampleFilterType;
    TransformType::Pointer finalTransform = TransformType::New();

  finalTransform->SetParameters( finalParameters );
  finalTransform->SetFixedParameters( transform->GetFixedParameters() );

  ResampleFilterType::Pointer resample = ResampleFilterType::New();

  resample->SetTransform( finalTransform );
  resample->SetInput( movingImageReader->GetOutput() );

  FixedImageType::Pointer fixedImage = fixedImageReader->GetOutput();

  PixelType backgroundGrayLevel = 100;

  resample->SetSize(    fixedImage->GetLargestPossibleRegion().GetSize() );
  resample->SetOutputOrigin(  fixedImage->GetOrigin() );
  resample->SetOutputSpacing( fixedImage->GetSpacing() );
  resample->SetOutputDirection( fixedImage->GetDirection() );
  resample->SetDefaultPixelValue( backgroundGrayLevel );

  typedef  unsigned char                           OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
  typedef itk::CastImageFilter<
                        FixedImageType,
                        OutputImageType >          CastFilterType;
  typedef itk::ImageFileWriter< OutputImageType >  WriterType;

  WriterType::Pointer      writer =  WriterType::New();
  CastFilterType::Pointer  caster =  CastFilterType::New();

  writer->SetFileName(argv[3]);

  caster->SetInput( resample->GetOutput() );
  writer->SetInput( caster->GetOutput()   );
  writer->Update();

  typedef itk::CheckerBoardImageFilter< FixedImageType > CheckerBoardFilterType;

  CheckerBoardFilterType::Pointer checker = CheckerBoardFilterType::New();

  checker->SetInput1( fixedImage );
  checker->SetInput2( resample->GetOutput() );

  caster->SetInput( checker->GetOutput() );
  writer->SetInput( caster->GetOutput()   );

  resample->SetDefaultPixelValue( 0 );

  TransformType::Pointer identityTransform = TransformType::New();
  identityTransform->SetIdentity();
  resample->SetTransform( identityTransform );
  writer->SetFileName(argv[4]);
  writer->Update();

  resample->SetTransform( registration->GetTransform() );   
  writer->SetFileName(argv[5]);
  writer->Update();

  long t2 = GetTickCount(); //程序段结束后取得系统运行时间(ms) 
  cout << "总运行用时：" << (t2 - t1)*0.001 << "秒" << endl;
  cout << "总运行用时：" << (t2 - t1) / 60000 << "分钟" << endl;

  std::cout << std::endl << "Done！" << std::endl;
  system("pause");
  return EXIT_SUCCESS;
}
