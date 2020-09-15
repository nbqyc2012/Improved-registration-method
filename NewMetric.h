#ifndef NewMetric_h
#define NewMetric_h

#include "itkImageToImageMetric.h"
#include "itkPoint.h"
#include "itkIndex.h"
#include "itkBSplineDerivativeKernelFunction.h"
#include "itkArray2D.h"

#include "itkSimpleFastMutexLock.h"


namespace itk
{
template <typename TFixedImage, typename TMovingImage>
class ITK_TEMPLATE_EXPORT MattesMutualInformationImageToImageMetric:
  public ImageToImageMetric<TFixedImage, TMovingImage>
{
public:
  typedef MattesMutualInformationImageToImageMetric     Self;
  typedef ImageToImageMetric<TFixedImage, TMovingImage> Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  itkNewMacro(Self);
  itkTypeMacro(MattesMutualInformationImageToImageMetric,
               ImageToImageMetric);

  /** Types inherited from Superclass. */
  typedef typename Superclass::TransformType                  TransformType;
  typedef typename Superclass::TransformPointer               TransformPointer;
  typedef typename Superclass::TransformJacobianType          TransformJacobianType;
  typedef typename Superclass::InterpolatorType               InterpolatorType;
  typedef typename Superclass::MeasureType                    MeasureType;
  typedef typename Superclass::DerivativeType                 DerivativeType;
  typedef typename Superclass::ParametersType                 ParametersType;
  typedef typename Superclass::FixedImageType                 FixedImageType;
  typedef typename Superclass::MovingImageType                MovingImageType;
  typedef typename Superclass::MovingImagePointType           MovingImagePointType;
  typedef typename Superclass::FixedImageConstPointer         FixedImageConstPointer;
  typedef typename Superclass::MovingImageConstPointer        MovingImageConstPointer;
  typedef typename Superclass::BSplineTransformWeightsType    BSplineTransformWeightsType;
  typedef typename Superclass::BSplineTransformIndexArrayType BSplineTransformIndexArrayType;
  typedef typename Superclass::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass::FixedImageSampleContainer    FixedImageSampleContainer;
  typedef typename Superclass::ImageDerivativesType         ImageDerivativesType;
  typedef typename Superclass::WeightsValueType             WeightsValueType;
  typedef typename Superclass::IndexValueType               IndexValueType;
  typedef typename FixedImageType::OffsetValueType OffsetValueType;

  itkStaticConstMacro(MovingImageDimension, unsigned int,
                      MovingImageType::ImageDimension);
  virtual void Initialize(void) ITK_OVERRIDE;

  MeasureType GetValue(const ParametersType & parameters) const ITK_OVERRIDE;

  /*得到互信息测度导数*/
  void GetDerivative(const ParametersType & parameters, DerivativeType & Derivative) const ITK_OVERRIDE;

  /*获得单值优化器的值和导数。*/
  void GetValueAndDerivative(const ParametersType & parameters, MeasureType & Value, DerivativeType & Derivative) const ITK_OVERRIDE;

/*Parzen直方图中使用的bin数量为50由于需要填充，所以最小值为5用三次b样条内核窗口。*/
  itkSetClampMacro( NumberOfHistogramBins, SizeValueType,
                    5, NumericTraits<SizeValueType>::max() );
  itkGetConstReferenceMacro(NumberOfHistogramBins, SizeValueType);

  /*该变量选择用于计算相对于变换参数的度规导数的方法。有两种计算模式可供选择。两者之间的选择是计算速度和内存分配之间的权衡。*/
  itkSetMacro(UseExplicitPDFDerivatives, bool);
  itkGetConstReferenceMacro(UseExplicitPDFDerivatives, bool);
  itkBooleanMacro(UseExplicitPDFDerivatives);

  typedef double PDFValueType; //使用双精度变量提高准确性

  typedef Image<PDFValueType, 2> JointPDFType;
  typedef Image<PDFValueType, 3> JointPDFDerivativesType;

  /*获取创建度量值时使用的内部联合pdf图像。*/
  const typename JointPDFType::Pointer GetJointPDF () const
    {
    if( this->m_MMIMetricPerThreadVariables == ITK_NULLPTR )
      {
      return JointPDFType::Pointer(ITK_NULLPTR);
      }
    return this->m_MMIMetricPerThreadVariables[0].JointPDF;
    }

  /*创建度量导数值*/
  const typename JointPDFDerivativesType::Pointer GetJointPDFDerivatives () const
    {
    if( this->m_MMIMetricPerThreadVariables == ITK_NULLPTR )
      {
      return JointPDFDerivativesType::Pointer(ITK_NULLPTR);
      }
    return this->m_MMIMetricPerThreadVariables[0].JointPDFDerivatives;
    }

protected:

  MattesMutualInformationImageToImageMetric();
  virtual ~MattesMutualInformationImageToImageMetric() ITK_OVERRIDE;
  void PrintSelf(std::ostream & os, Indent indent) const ITK_OVERRIDE;

private:
  ITK_DISALLOW_COPY_AND_ASSIGN(MattesMutualInformationImageToImageMetric);

  typedef JointPDFType::IndexType             JointPDFIndexType;
  typedef JointPDFType::PixelType             JointPDFValueType;
  typedef JointPDFType::RegionType            JointPDFRegionType;
  typedef JointPDFType::SizeType              JointPDFSizeType;
  typedef JointPDFDerivativesType::IndexType  JointPDFDerivativesIndexType;
  typedef JointPDFDerivativesType::PixelType  JointPDFDerivativesValueType;
  typedef JointPDFDerivativesType::RegionType JointPDFDerivativesRegionType;
  typedef JointPDFDerivativesType::SizeType   JointPDFDerivativesSizeType;

  /** 定义b样条核函数和导数函数*/
  typedef BSplineKernelFunction<3,PDFValueType>           CubicBSplineFunctionType;
  typedef BSplineDerivativeKernelFunction<3,PDFValueType> CubicBSplineDerivativeFunctionType;
  void ComputeFixedImageParzenWindowIndices( FixedImageSampleContainer & samples);

  /*计算每个参数的PDF导数。*/
  void ComputePDFDerivatives(ThreadIdType threadId, unsigned int sampleNumber, int movingImageParzenWindowIndex,
                                     const ImageDerivativesType
                                     &  movingImageGradientValue,
                                     PDFValueType cubicBSplineDerivativeValue) const;

  virtual void GetValueThreadPreProcess(ThreadIdType threadId, bool withinSampleThread) const ITK_OVERRIDE;
  virtual void GetValueThreadPostProcess(ThreadIdType threadId, bool withinSampleThread) const ITK_OVERRIDE;
  virtual bool GetValueThreadProcessSample(ThreadIdType threadId, SizeValueType fixedImageSample,
                                                  const MovingImagePointType & mappedPoint,
                                                  double movingImageValue) const ITK_OVERRIDE;

  virtual void GetValueAndDerivativeThreadPreProcess( ThreadIdType threadId, bool withinSampleThread) const ITK_OVERRIDE;
  virtual void GetValueAndDerivativeThreadPostProcess( ThreadIdType threadId, bool withinSampleThread) const ITK_OVERRIDE;
  virtual bool GetValueAndDerivativeThreadProcessSample(ThreadIdType threadId, SizeValueType fixedImageSample,
                                                               const MovingImagePointType & mappedPoint,
                                                               double movingImageValue, const ImageDerivativesType &
                                                               movingImageGradientValue) const ITK_OVERRIDE;

  /*定义变量的边缘和联合直方图。*/
  SizeValueType m_NumberOfHistogramBins;
  PDFValueType  m_MovingImageNormalizedMin;
  PDFValueType  m_FixedImageNormalizedMin;
  PDFValueType  m_FixedImageTrueMin;
  PDFValueType  m_FixedImageTrueMax;
  PDFValueType  m_MovingImageTrueMin;
  PDFValueType  m_MovingImageTrueMax;
  PDFValueType  m_FixedImageBinSize;
  PDFValueType  m_MovingImageBinSize;

  /*计算Parzen直方图的三次b样条核。*/
  typename CubicBSplineFunctionType::Pointer           m_CubicBSplineKernel;
  typename CubicBSplineDerivativeFunctionType::Pointer m_CubicBSplineDerivativeKernel;

  /*用于存储PDF比率值的数组 */
  typedef PDFValueType        PRatioType;
  typedef Array2D<PRatioType> PRatioArrayType;

  mutable PRatioArrayType m_PRatioArray;

  /*移动图像的边缘PDF。*/
  typedef std::vector< PDFValueType > MarginalPDFType;
  mutable MarginalPDFType             m_MovingImageMarginalPDF;

  struct MMIMetricPerThreadStruct
  {
    int JointPDFStartBin;
    int JointPDFEndBin;

    PDFValueType JointPDFSum;

    /*辅助变量用于累积测度导数。 */
    DerivativeType MetricDerivative;

    typename JointPDFType::Pointer            JointPDF;
    typename JointPDFDerivativesType::Pointer JointPDFDerivatives;

    typename TransformType::JacobianType Jacobian;

    MarginalPDFType FixedImageMarginalPDF;
  };

#if !defined(ITK_WRAPPING_PARSER)
  itkPadStruct( ITK_CACHE_LINE_ALIGNMENT, MMIMetricPerThreadStruct,
                                            PaddedMMIMetricPerThreadStruct);
  itkAlignedTypedef( ITK_CACHE_LINE_ALIGNMENT, PaddedMMIMetricPerThreadStruct,
                                               AlignedMMIMetricPerThreadStruct );

  mutable AlignedMMIMetricPerThreadStruct * m_MMIMetricPerThreadVariables;
#endif

  bool         m_UseExplicitPDFDerivatives;
  mutable bool m_ImplicitDerivativesSecondPass;
};
} 

#ifndef ITK_MANUAL_INSTANTIATION
#include "NewMetric.hxx"
#endif

#endif
