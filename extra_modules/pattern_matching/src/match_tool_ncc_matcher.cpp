// match_tool_ncc_matcher.cpp — Grayscale 8U pyramid + SIMD CCORR (MatchTool), NCC denominator from MatchToolDlg
#include "match_tool_ncc_matcher.h"
#include "opencvsharp_license_guard.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <opencv2/imgproc.hpp>

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define MATCHTOOL_HAS_SSE2 1
#include <emmintrin.h>
#endif


#define MATCHTOOL_HAS_AVX2 1
#include <immintrin.h>


#define D2R (CV_PI / 180.0)
#define R2D (180.0 / CV_PI)
#define VISION_TOLERANCE 0.0000001
#define MATCH_CANDIDATE_NUM 5
// Template-area threshold above which we delegate the per-row spatial CCORR
// to cv::matchTemplate (which uses FFT internally for large inputs).
#define MATCHTOOL_FFT_AREA_THRESHOLD 10000

namespace cv {
namespace pattern_matching {

namespace {

#ifdef MATCHTOOL_HAS_SSE2
inline int hsum_epi32_sse2(__m128i x) {
    __m128i hi64 = _mm_unpackhi_epi64(x, x);
    __m128i sum64 = _mm_add_epi32(hi64, x);
    __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i sum32 = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);
}

inline int32_t IM_Conv_SIMD(const uint8_t* pCharKernel, const uint8_t* pCharConv, int iLength) {
    const int iBlockSize = 16;
    const int Block = iLength / iBlockSize;
    __m128i SumV = _mm_setzero_si128();
    for (int Y = 0; Y < Block * iBlockSize; Y += iBlockSize) {
        __m128i SrcK = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pCharKernel + Y));
        __m128i SrcC = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pCharConv + Y));
        __m128i SrcK_L = _mm_unpacklo_epi8(SrcK, _mm_setzero_si128());
        __m128i SrcK_H = _mm_unpackhi_epi8(SrcK, _mm_setzero_si128());
        __m128i SrcC_L = _mm_unpacklo_epi8(SrcC, _mm_setzero_si128());
        __m128i SrcC_H = _mm_unpackhi_epi8(SrcC, _mm_setzero_si128());
        __m128i SumL = _mm_madd_epi16(SrcK_L, SrcC_L);
        __m128i SumH = _mm_madd_epi16(SrcK_H, SrcC_H);
        __m128i SumT = _mm_add_epi32(SumL, SumH);
        SumV = _mm_add_epi32(SumV, SumT);
    }
    int32_t Sum = hsum_epi32_sse2(SumV);
    for (int Y = Block * iBlockSize; Y < iLength; Y++)
        Sum += pCharKernel[Y] * pCharConv[Y];
    return Sum;
}
#endif

#ifdef MATCHTOOL_HAS_AVX2
inline int hsum_epi32_avx2(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    return hsum_epi32_sse2(_mm_add_epi32(lo, hi));
}

inline int32_t IM_Conv_AVX2(const uint8_t* pK, const uint8_t* pC, int iLength) {
    const int iBlockSize = 32;
    const int Block = iLength / iBlockSize;
    __m256i SumV = _mm256_setzero_si256();
    const __m256i zero256 = _mm256_setzero_si256();
    for (int Y = 0; Y < Block * iBlockSize; Y += iBlockSize) {
        __m256i K = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pK + Y));
        __m256i C = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pC + Y));
        __m256i K_lo = _mm256_unpacklo_epi8(K, zero256);
        __m256i K_hi = _mm256_unpackhi_epi8(K, zero256);
        __m256i C_lo = _mm256_unpacklo_epi8(C, zero256);
        __m256i C_hi = _mm256_unpackhi_epi8(C, zero256);
        __m256i P_lo = _mm256_madd_epi16(K_lo, C_lo);
        __m256i P_hi = _mm256_madd_epi16(K_hi, C_hi);
        SumV = _mm256_add_epi32(SumV, _mm256_add_epi32(P_lo, P_hi));
    }
    int32_t Sum = hsum_epi32_avx2(SumV);
    int tail = Block * iBlockSize;
    // 16-byte SSE2 tail
    if (iLength - tail >= 16) {
        __m128i SrcK = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pK + tail));
        __m128i SrcC = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pC + tail));
        __m128i zero128 = _mm_setzero_si128();
        __m128i KL = _mm_unpacklo_epi8(SrcK, zero128);
        __m128i KH = _mm_unpackhi_epi8(SrcK, zero128);
        __m128i CL = _mm_unpacklo_epi8(SrcC, zero128);
        __m128i CH = _mm_unpackhi_epi8(SrcC, zero128);
        Sum += hsum_epi32_sse2(_mm_add_epi32(_mm_madd_epi16(KL, CL), _mm_madd_epi16(KH, CH)));
        tail += 16;
    }
    for (int Y = tail; Y < iLength; Y++)
        Sum += pK[Y] * pC[Y];
    return Sum;
}
#endif

void simdCCORR_8u(const cv::Mat& matSrc, const cv::Mat& matTemplate, cv::Mat& matResult) {
#if defined(MATCHTOOL_HAS_SSE2) || defined(MATCHTOOL_HAS_AVX2)
    matResult.create(matSrc.rows - matTemplate.rows + 1,
        matSrc.cols - matTemplate.cols + 1, CV_32FC1);
    matResult.setTo(0);
    const int t_r_end = matTemplate.rows;
    const int tw = matTemplate.cols;
    const int srcStep = (int)matSrc.step[0];
    const int tplStep = (int)matTemplate.step[0];
#ifdef MATCHTOOL_HAS_AVX2
    // Below ~64 bytes/row the AVX2 horizontal-sum overhead per call cancels
    // the gain from the 32-byte vector. SSE2 wins for narrow templates.
    const bool useAvx2 = (tw >= 64);
#endif
    for (int r = 0; r < matResult.rows; r++) {
        float* r_matResult = matResult.ptr<float>(r);
        const uchar* r_source = matSrc.ptr<uchar>(r);
        for (int c = 0; c < matResult.cols; ++c, ++r_matResult, ++r_source) {
            const uchar* r_template = matTemplate.ptr<uchar>();
            const uchar* r_sub_source = r_source;
            float acc = 0.f;
#ifdef MATCHTOOL_HAS_AVX2
            if (useAvx2) {
                for (int t_r = 0; t_r < t_r_end; ++t_r, r_sub_source += srcStep, r_template += tplStep) {
                    acc += static_cast<float>(IM_Conv_AVX2(r_template, r_sub_source, tw));
                }
            }
            else
#endif
            {
                for (int t_r = 0; t_r < t_r_end; ++t_r, r_sub_source += srcStep, r_template += tplStep) {
                    acc += static_cast<float>(IM_Conv_SIMD(r_template, r_sub_source, tw));
                }
            }
            *r_matResult = acc;
        }
    }
#else
    cv::matchTemplate(matSrc, matTemplate, matResult, cv::TM_CCORR);
#endif
}

// MatchToolDlg::CCOEFF_Denominator (single-channel 8U source).
// If `sumIn` and `sqsumIn` are non-empty, they are used as precomputed integrals.
// `integralOffset` translates output coordinates into the integral coordinate
// system, allowing reuse of an integral computed over a LARGER source than
// matSrc (e.g. once per pyramid level, then queried per ROI). The default
// (0,0) preserves the original behaviour where the integral matches matSrc.
void CCOEFF_Denominator_8u(
    const cv::Mat& matSrc,
    const cv::Mat& templ,
    const cv::Scalar& templMean,
    double templNorm,
    double invArea,
    bool resultEqual1,
    cv::Mat& matResult,
    const cv::Mat& sumIn = cv::Mat(),
    const cv::Mat& sqsumIn = cv::Mat(),
    cv::Point integralOffset = cv::Point(0, 0))
{
    if (resultEqual1 || templNorm < DBL_EPSILON) {
        matResult.setTo(1.0f);
        return;
    }

    const int th = templ.rows;
    const int tw = templ.cols;

    cv::Mat sum, sqsum;
    if (!sumIn.empty() && !sqsumIn.empty()) {
        sum = sumIn;
        sqsum = sqsumIn;
    }
    else {
        cv::integral(matSrc, sum, sqsum, CV_64F);
    }

    double* q0 = reinterpret_cast<double*>(sqsum.data);
    double* q1 = q0 + tw;
    double* q2 = reinterpret_cast<double*>(sqsum.data + th * sqsum.step);
    double* q3 = q2 + tw;

    double* p0 = reinterpret_cast<double*>(sum.data);
    double* p1 = p0 + tw;
    double* p2 = reinterpret_cast<double*>(sum.data + th * sum.step);
    double* p3 = p2 + tw;

    const int sumstep = sum.data ? static_cast<int>(sum.step / sizeof(double)) : 0;
    const int sqstep = sqsum.data ? static_cast<int>(sqsum.step / sizeof(double)) : 0;

    const double dTemplMean0 = templMean[0];
    const int offX = integralOffset.x;
    const int offY = integralOffset.y;

    for (int i = 0; i < matResult.rows; i++) {
        float* rrow = matResult.ptr<float>(i);
        int idx = (i + offY) * sumstep + offX;
        int idx2 = (i + offY) * sqstep + offX;

        for (int j = 0; j < matResult.cols; j++, idx++, idx2++) {
            double num = rrow[j];
            double t;

            t = p0[idx] - p1[idx] - p2[idx] + p3[idx];
            double wndMean2 = t * t * invArea;
            num -= t * dTemplMean0;

            t = q0[idx2] - q1[idx2] - q2[idx2] + q3[idx2];
            double wndSum2 = t;

            double diff2 = std::max(wndSum2 - wndMean2, 0.0);
            double denom_t;
            if (diff2 <= std::min(0.5, 10.0 * FLT_EPSILON * wndSum2))
                denom_t = 0;
            else
                denom_t = std::sqrt(diff2) * templNorm;

            double score;
            if (denom_t > DBL_EPSILON) {
                score = num / denom_t;
                if (score > 1.0) score = 1.0;
                else if (score < -1.0) score = -1.0;
            }
            else {
                score = 0.0;
            }

            rrow[j] = (float)score;
        }
    }
}

void matchTemplateMatchTool8u(
    const cv::Mat& matSrc,
    const cv::Mat& templ,
    const cv::Scalar& templMean,
    double templNorm,
    double invArea,
    bool resultEqual1,
    cv::Mat& matResult,
    const cv::Mat& sumIn = cv::Mat(),
    const cv::Mat& sqsumIn = cv::Mat())
{
    if (matSrc.rows < templ.rows || matSrc.cols < templ.cols) {
        matResult.release();
        return;
    }

    // FFT helps when there's enough output to amortize its setup AND enough
    // template work for FFT savings to dominate. For refinement (output ~7x7),
    // setup overhead beats FFT savings — spatial wins. We require BOTH the
    // template area and the output area to be large enough.
    const int outCols = matSrc.cols - templ.cols + 1;
    const int outRows = matSrc.rows - templ.rows + 1;
    const int templArea = templ.cols * templ.rows;
    const int outArea = outCols * outRows;
    if (templArea >= MATCHTOOL_FFT_AREA_THRESHOLD &&
        outArea >= MATCHTOOL_FFT_AREA_THRESHOLD) {
        if (resultEqual1 || templNorm < DBL_EPSILON) {
            matResult.create(outRows, outCols, CV_32FC1);
            matResult.setTo(1.0f);
            return;
        }
        cv::matchTemplate(matSrc, templ, matResult, cv::TM_CCOEFF_NORMED);
        return;
    }

    simdCCORR_8u(matSrc, templ, matResult);
    CCOEFF_Denominator_8u(matSrc, templ, templMean, templNorm, invArea, resultEqual1,
        matResult, sumIn, sqsumIn);
}

static cv::Point2f ptRotatePt2f(cv::Point2f ptInput, cv::Point2f ptOrg, double dAngle) {
    double dWidth = ptOrg.x * 2;
    double dHeight = ptOrg.y * 2;
    double dY1 = dHeight - ptInput.y;
    double dY2 = dHeight - ptOrg.y;

    double dX = (ptInput.x - ptOrg.x) * cos(dAngle) - (dY1 - ptOrg.y) * sin(dAngle) + ptOrg.x;
    double dY = (ptInput.x - ptOrg.x) * sin(dAngle) + (dY1 - ptOrg.y) * cos(dAngle) + dY2;

    dY = -dY + dHeight;
    return cv::Point2f((float)dX, (float)dY);
}

static cv::Size GetBestRotationSize(cv::Size sizeSrc, cv::Size sizeDst, double dRAngle) {
    double dRAngle_radian = dRAngle * D2R;
    cv::Point ptLT(0, 0), ptLB(0, sizeSrc.height - 1);
    cv::Point ptRB(sizeSrc.width - 1, sizeSrc.height - 1), ptRT(sizeSrc.width - 1, 0);
    cv::Point2f ptCenter((sizeSrc.width - 1) / 2.0f, (sizeSrc.height - 1) / 2.0f);

    cv::Point2f ptLT_R = ptRotatePt2f(cv::Point2f(ptLT), ptCenter, dRAngle_radian);
    cv::Point2f ptLB_R = ptRotatePt2f(cv::Point2f(ptLB), ptCenter, dRAngle_radian);
    cv::Point2f ptRB_R = ptRotatePt2f(cv::Point2f(ptRB), ptCenter, dRAngle_radian);
    cv::Point2f ptRT_R = ptRotatePt2f(cv::Point2f(ptRT), ptCenter, dRAngle_radian);

    float fTopY = std::max(std::max(ptLT_R.y, ptLB_R.y), std::max(ptRB_R.y, ptRT_R.y));
    float fBottomY = std::min(std::min(ptLT_R.y, ptLB_R.y), std::min(ptRB_R.y, ptRT_R.y));
    float fRightX = std::max(std::max(ptLT_R.x, ptLB_R.x), std::max(ptRB_R.x, ptRT_R.x));
    float fLeftX = std::min(std::min(ptLT_R.x, ptLB_R.x), std::min(ptRB_R.x, ptRT_R.x));

    double dAngle = dRAngle;
    if (dAngle > 360) dAngle -= 360;
    else if (dAngle < 0) dAngle += 360;

    if (std::abs(std::abs(dAngle) - 90) < VISION_TOLERANCE ||
        std::abs(std::abs(dAngle) - 270) < VISION_TOLERANCE) {
        return cv::Size(sizeSrc.height, sizeSrc.width);
    }
    else if (std::abs(dAngle) < VISION_TOLERANCE ||
        std::abs(std::abs(dAngle) - 180) < VISION_TOLERANCE) {
        return sizeSrc;
    }

    if (dAngle > 90 && dAngle < 180) dAngle -= 90;
    else if (dAngle > 180 && dAngle < 270) dAngle -= 180;
    else if (dAngle > 270 && dAngle < 360) dAngle -= 270;

    const double dAngRad = dAngle * D2R;
    const float fH1 = static_cast<float>(sizeDst.width * sin(dAngRad) * cos(dAngRad));
    const float fH2 = static_cast<float>(sizeDst.height * sin(dAngRad) * cos(dAngRad));

    int iHalfHeight = static_cast<int>(ceil(static_cast<double>(fTopY - ptCenter.y - fH1)));
    int iHalfWidth = static_cast<int>(ceil(static_cast<double>(fRightX - ptCenter.x - fH2)));

    cv::Size sizeRet(iHalfWidth * 2, iHalfHeight * 2);

    bool bWrongSize = (sizeDst.width < sizeRet.width && sizeDst.height > sizeRet.height) ||
        (sizeDst.width > sizeRet.width && sizeDst.height < sizeRet.height) ||
        (sizeDst.area() > sizeRet.area());

    if (bWrongSize)
        sizeRet = cv::Size(static_cast<int>(fRightX - fLeftX + 0.5f), static_cast<int>(fTopY - fBottomY + 0.5f));

    return sizeRet;
}

// MatchToolDlg::SubPixEsimation — quadratic fit over 3 angles × 3×3 correlation samples
static bool matchToolSubPixelEstimation(
    const float vecScore[3][3][3],
    int iMaxScoreIndex,
    double dAngleStep,
    double dX_maxScore,
    double dY_maxScore,
    double dTheta_maxScore_deg,
    double* dNewX,
    double* dNewY,
    double* dNewAngleDeg)
{
    cv::Mat matA(27, 10, CV_64F);
    cv::Mat matS(27, 1, CV_64F);
    int iRow = 0;
    for (int theta = 0; theta <= 2; theta++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                const int hypIdx = iMaxScoreIndex + (theta - 1);
                const double dX = dX_maxScore + x;
                const double dY = dY_maxScore + y;
                const double dT = (dTheta_maxScore_deg + (theta - 1) * dAngleStep) * D2R;
                matA.at<double>(iRow, 0) = dX * dX;
                matA.at<double>(iRow, 1) = dY * dY;
                matA.at<double>(iRow, 2) = dT * dT;
                matA.at<double>(iRow, 3) = dX * dY;
                matA.at<double>(iRow, 4) = dX * dT;
                matA.at<double>(iRow, 5) = dY * dT;
                matA.at<double>(iRow, 6) = dX;
                matA.at<double>(iRow, 7) = dY;
                matA.at<double>(iRow, 8) = dT;
                matA.at<double>(iRow, 9) = 1.0;
                matS.at<double>(iRow, 0) = vecScore[hypIdx][x + 1][y + 1];
                iRow++;
            }
        }
    }

    try {
        cv::Mat matAtA = matA.t() * matA;
        cv::Mat matZ = matAtA.inv() * matA.t() * matS;
        cv::Mat matZ_t;
        cv::transpose(matZ, matZ_t);
        double* dZ = matZ_t.ptr<double>(0);
        cv::Mat matK1 = (cv::Mat_<double>(3, 3) <<
            (2 * dZ[0]), dZ[3], dZ[4],
            dZ[3], (2 * dZ[1]), dZ[5],
            dZ[4], dZ[5], (2 * dZ[2]));
        cv::Mat matK2 = (cv::Mat_<double>(3, 1) << -dZ[6], -dZ[7], -dZ[8]);
        cv::Mat matDelta;
        cv::solve(matK1, matK2, matDelta, cv::DECOMP_LU);
        *dNewX = matDelta.at<double>(0, 0);
        *dNewY = matDelta.at<double>(1, 0);
        *dNewAngleDeg = matDelta.at<double>(2, 0) * R2D;
        return std::isfinite(*dNewX) && std::isfinite(*dNewY) && std::isfinite(*dNewAngleDeg);
    }
    catch (...) {
        return false;
    }
}

static cv::Point GetNextMaxLoc(
    cv::Mat& matResult,
    const cv::Point& ptMaxLoc,
    const cv::Size& sizeTemplate,
    double& dMaxValue,
    double dMaxOverlap)
{
    const int iStartX = cvRound(ptMaxLoc.x - sizeTemplate.width * (1.0 - dMaxOverlap));
    const int iStartY = cvRound(ptMaxLoc.y - sizeTemplate.height * (1.0 - dMaxOverlap));
    const int rw = cvRound(2.0 * sizeTemplate.width * (1.0 - dMaxOverlap));
    const int rh = cvRound(2.0 * sizeTemplate.height * (1.0 - dMaxOverlap));

    cv::rectangle(matResult,
        cv::Rect(iStartX, iStartY, rw, rh),
        cv::Scalar(-1), cv::FILLED);

    cv::Point ptNew;
    cv::minMaxLoc(matResult, nullptr, &dMaxValue, nullptr, &ptNew);
    return ptNew;
}

} // namespace

cv::Ptr<MatchToolNCCMatcher> MatchToolNCCMatcher::create() {
    return cv::makePtr<MatchToolNCCMatcherImpl>();
}

std::vector<MatchInfo> MatchToolNCCMatcher::MatchTemplate(
    cv::InputArray source,
    cv::InputArray templateImg,
    int numLevels,
    double scoreThreshold,
    double angleRange,
    double maxOverlap,
    int minReducedArea,
    int maxTargets,
    bool useSubPixel)
{
    cv::Ptr<MatchToolNCCMatcher> matcher = MatchToolNCCMatcher::create();
    matcher->setAngleRange(-angleRange, angleRange);
    if (!matcher->setTemplate(templateImg))
        return {};
    if (!matcher->train(numLevels, minReducedArea))
        return {};
    matcher->setScoreThreshold(scoreThreshold);
    matcher->setMaxOverlap(maxOverlap);
    matcher->setUseSubPixel(useSubPixel);
    return matcher->match(source, numLevels, maxTargets, 0);
}

MatchToolNCCMatcherImpl::MatchToolNCCMatcherImpl()
    : m_isTemplateSet(false)
    , m_minAngle(-5.0)
    , m_maxAngle(5.0)
    , m_scoreThreshold(0.7)
    , m_maxOverlap(0.5)
    , num_levels(0)
    , num_levels_for_src(0)
    , m_isLearned(false)
    , m_borderColor(0)
    , m_useSubPixel(false)
    , m_rotCacheMinAngle(0.0)
    , m_rotCacheMaxAngle(0.0)
    , m_rotCacheValid(false)
{
}

MatchToolNCCMatcherImpl::~MatchToolNCCMatcherImpl() {}

void MatchToolNCCMatcherImpl::invalidateRotationCache() {
    m_rotCache.clear();
    m_rotCacheAngles.clear();
    m_rotCacheStep.clear();
    m_rotCacheMinAngle = 0.0;
    m_rotCacheMaxAngle = 0.0;
    m_rotCacheValid = false;
}

void MatchToolNCCMatcherImpl::buildRotationCache() {
    m_rotCache.clear();
    m_rotCacheAngles.clear();
    m_rotCacheStep.clear();
    m_rotCacheValid = false;

    if (m_templatePyramid.empty() ||
        m_vecTemplMean.size() != m_templatePyramid.size())
        return;

    const int numLevels = static_cast<int>(m_templatePyramid.size());
    m_rotCache.resize(numLevels);
    m_rotCacheAngles.resize(numLevels);
    m_rotCacheStep.resize(numLevels, 0.0);

    const bool onlyOne = std::abs(m_maxAngle - m_minAngle) < VISION_TOLERANCE;

    for (int level = 0; level < numLevels; level++) {
        const cv::Mat& templ = m_templatePyramid[level];
        if (templ.empty())
            continue;

        const cv::Point2f templCenter((templ.cols - 1) / 2.0f, (templ.rows - 1) / 2.0f);
        const double meanVal = m_vecTemplMean[level][0];
        const int maxDim = std::max(templ.cols, templ.rows);
        const double step = std::atan(2.0 / maxDim) * R2D;
        m_rotCacheStep[level] = step;

        std::vector<double>& angles = m_rotCacheAngles[level];
        if (onlyOne) {
            angles.push_back(m_minAngle);
        }
        else {
            for (double a = m_minAngle; a <= m_maxAngle; a += step)
                angles.push_back(a);
            if (angles.empty() || std::abs(angles.back() - m_maxAngle) > VISION_TOLERANCE)
                angles.push_back(m_maxAngle);
        }

        m_rotCache[level].resize(angles.size());

        cv::parallel_for_(cv::Range(0, (int)angles.size()), [&, level](const cv::Range& range) {
            for (int i = range.start; i < range.end; i++) {
                const double angle = angles[i];
                const cv::Size sizeRotT = GetBestRotationSize(templ.size(), templ.size(), angle);

                const float fTxT = (sizeRotT.width - 1) / 2.0f - templCenter.x;
                const float fTyT = (sizeRotT.height - 1) / 2.0f - templCenter.y;

                cv::Mat rMat = cv::getRotationMatrix2D(templCenter, angle, 1.0);
                rMat.at<double>(0, 2) += fTxT;
                rMat.at<double>(1, 2) += fTyT;

                cv::Mat rotT;
                cv::warpAffine(
                    templ, rotT, rMat, sizeRotT,
                    cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(meanVal));

                if (rotT.empty())
                    continue;

                const double invArea = 1.0 / (rotT.rows * rotT.cols);
                cv::Scalar mean, stddev;
                cv::meanStdDev(rotT, mean, stddev);
                double norm = stddev[0] * stddev[0];
                const bool eq1 = (norm < DBL_EPSILON);
                norm = std::sqrt(norm) / std::sqrt(invArea);

                RotatedCacheEntry& e = m_rotCache[level][i];
                e.templ = rotT;
                e.mean = mean;
                e.norm = norm;
                e.invArea = invArea;
                e.resultEqual1 = eq1;
            }
        });
    }

    m_rotCacheMinAngle = m_minAngle;
    m_rotCacheMaxAngle = m_maxAngle;
    m_rotCacheValid = true;
}

int MatchToolNCCMatcherImpl::findNearestCachedAngle(int level, double angle) const {
    if (level < 0 || level >= static_cast<int>(m_rotCacheAngles.size()))
        return -1;
    const std::vector<double>& angles = m_rotCacheAngles[level];
    if (angles.empty())
        return -1;
    if (angles.size() == 1)
        return 0;

    const double step = m_rotCacheStep[level];
    int idx = (step > 0.0)
        ? static_cast<int>(std::round((angle - angles[0]) / step))
        : 0;
    if (idx < 0) idx = 0;
    const int last = static_cast<int>(angles.size()) - 1;
    if (idx > last) idx = last;

    // Guard against rounding edge cases vs neighbors.
    if (idx > 0 && std::abs(angles[idx - 1] - angle) < std::abs(angles[idx] - angle))
        idx--;
    if (idx < last && std::abs(angles[idx + 1] - angle) < std::abs(angles[idx] - angle))
        idx++;
    return idx;
}

bool MatchToolNCCMatcherImpl::setTemplate(cv::InputArray templateImg) {
    cv::Mat temp = templateImg.getMat();
    if (temp.empty())
        return false;

    if (temp.channels() == 1)
        m_templateGray8u = temp.clone();
    else if (temp.channels() == 3)
        cv::cvtColor(temp, m_templateGray8u, cv::COLOR_BGR2GRAY);
    else if (temp.channels() == 4)
        cv::cvtColor(temp, m_templateGray8u, cv::COLOR_BGRA2GRAY);
    else
        return false;

    double meanVal = cv::mean(m_templateGray8u)[0];
    m_borderColor = meanVal < 128 ? 255 : 0;

    m_isTemplateSet = true;
    m_templatePyramid.clear();
    m_isLearned = false;
    invalidateRotationCache();
    return true;
}

bool MatchToolNCCMatcherImpl::train(int numLevels, int minReducedArea) {
    if (!m_isTemplateSet)
        return false;

    buildPyramid(m_templateGray8u, m_templatePyramid, numLevels, minReducedArea);

    const int n = (int)m_templatePyramid.size();
    m_vecTemplMean.resize(n);
    m_vecTemplNorm.resize(n);
    m_vecInvArea.resize(n);
    m_vecResultEqual1.resize(n, false);

    for (int i = 0; i < n; i++) {
        const cv::Mat& T = m_templatePyramid[i];
        const double invArea = 1.0 / (T.rows * T.cols);
        cv::Scalar mean, stddev;
        cv::meanStdDev(T, mean, stddev);

        double templNorm =
            stddev[0] * stddev[0] + stddev[1] * stddev[1] +
            stddev[2] * stddev[2] + stddev[3] * stddev[3];

        if (templNorm < DBL_EPSILON)
            m_vecResultEqual1[i] = true;

        templNorm = std::sqrt(templNorm);
        templNorm /= std::sqrt(invArea);

        m_vecInvArea[i] = invArea;
        m_vecTemplMean[i] = mean;
        m_vecTemplNorm[i] = templNorm;
    }

    m_isLearned = true;

    // Pre-compute rotated templates (angles fixed until the next setTemplate).
    buildRotationCache();

    return true;
}

void MatchToolNCCMatcherImpl::buildPyramid(const cv::Mat& image, std::vector<cv::Mat>& pyramid, int numLevels, int minArea) {
    pyramid.clear();
    if (image.empty())
        return;

    cv::Mat current = image.clone();
    pyramid.push_back(current);

    int currentArea = current.rows * current.cols;
    num_levels = 0;

    for (int i = 0; i < numLevels; i++) {
        if (currentArea <= minArea)
            break;

        cv::Mat down;
        cv::pyrDown(current, down);
        if (down.empty() || down.rows < 4 || down.cols < 4)
            break;

        pyramid.push_back(down);
        current = down;
        currentArea = current.rows * current.cols;
    }

    num_levels = (int)pyramid.size();
}

void MatchToolNCCMatcherImpl::buildSrcPyramid(const cv::Mat& image, std::vector<cv::Mat>& pyramid, int topLayer) {
    pyramid.clear();
    if (image.empty())
        return;

    cv::Mat current = image.clone();
    pyramid.push_back(current);
    for (int i = 0; i < topLayer; i++) {
        cv::Mat down;
        cv::pyrDown(current, down);
        pyramid.push_back(down);
        current = down;
    }
}

std::vector<MatchInfo> MatchToolNCCMatcherImpl::match(
    cv::InputArray source,
    int numLevels,
    int maxTargets,
    int minPyramidLevel)
{
    if (!opencvsharp_license_runtime_activated())
        return {};

    cv::Mat src = source.getMat();
    if (src.empty() || m_templatePyramid.empty())
        return {};

    if ((m_templateGray8u.cols < src.cols && m_templateGray8u.rows > src.rows) ||
        (m_templateGray8u.cols > src.cols && m_templateGray8u.rows < src.rows)) {
        return {};
    }

    cv::Mat srcGray;
    if (src.channels() == 1)
        srcGray = src.clone();
    else if (src.channels() == 3)
        cv::cvtColor(src, srcGray, cv::COLOR_BGR2GRAY);
    else if (src.channels() == 4)
        cv::cvtColor(src, srcGray, cv::COLOR_BGRA2GRAY);
    else
        return {};

    std::vector<cv::Mat> srcPyramid;
    buildSrcPyramid(srcGray, srcPyramid, static_cast<int>(m_templatePyramid.size()) - 1);

    if (srcPyramid.empty())
        return {};

    int topLevel = std::min(
        static_cast<int>(m_templatePyramid.size()),
        static_cast<int>(srcPyramid.size())) - 1;

    topLevel = std::min(topLevel, numLevels);
    num_levels_for_src = topLevel;

    if (topLevel < 0)
        return {};

    int iTopLayer = topLevel;
    std::vector<double> vecLayerScore(iTopLayer + 1);
    vecLayerScore[0] = m_scoreThreshold;
    for (int i = 1; i <= iTopLayer; i++)
        vecLayerScore[i] = vecLayerScore[i - 1] * 0.9;
    // Template-rotation at top level uses an axis-aligned bbox that includes
    // some source background outside the rotated rectangle. This biases the
    // CCOEFF denominator (typically downward). Loosen the top-level threshold
    // to keep recall; refinement re-scores accurately using source-rotation.
    const double kTopLevelBiasFactor = 0.85;
    vecLayerScore[iTopLayer] *= kTopLevelBiasFactor;

    int iTopSrcW = srcPyramid[topLevel].cols;
    int iTopSrcH = srcPyramid[topLevel].rows;
    cv::Point2f ptCenter((iTopSrcW - 1) / 2.0f, (iTopSrcH - 1) / 2.0f);

    float m_maxOverlap_topLevel = (float)m_maxOverlap;
    if (0 < m_maxOverlap && m_maxOverlap <= 0.3)
        m_maxOverlap_topLevel = (float)std::min(std::pow(1.2, topLevel) * m_maxOverlap, 1.0);
    else if (m_maxOverlap == 0)
        m_maxOverlap_topLevel = (float)std::min(0.05 * topLevel, 1.0);

    struct MatchCandidate {
        cv::Point2f pt;
        double score;
        double angle;
    };

    // Rotation cache is tied to the template (cleared in setTemplate, built in
    // train). setAngleRange does not rebuild it — set angle range before train.
    if (!m_rotCacheValid || m_rotCache.size() != m_templatePyramid.size()) {
        if (!m_isLearned)
            return {};
        buildRotationCache();
    }

    const cv::Mat& templTop = m_templatePyramid[topLevel];
    if (topLevel >= static_cast<int>(m_rotCache.size()) || m_rotCache[topLevel].empty())
        return {};
    const std::vector<RotatedCacheEntry>& topCacheEntries = m_rotCache[topLevel];
    const std::vector<double>& vecAngles = m_rotCacheAngles[topLevel];
    if (vecAngles.empty())
        return {};
    const bool only_search_angle = (vecAngles.size() == 1);

    std::vector<MatchCandidate> vecMatchParameter;
    std::mutex topLevelMtx;

    const cv::Mat& srcTop = srcPyramid[topLevel];

    // Source integral is constant across all angles under template rotation —
    // compute once and reuse via the optional precomputed-integral overload.
    cv::Mat srcTopSum, srcTopSqsum;
    cv::integral(srcTop, srcTopSum, srcTopSqsum, CV_64F);

    cv::parallel_for_(cv::Range(0, (int)vecAngles.size()), [&](const cv::Range& range) {
        std::vector<MatchCandidate> localMatches;
        localMatches.reserve((size_t)(range.end - range.start) * (maxTargets + MATCH_CANDIDATE_NUM));

        for (int i = range.start; i < range.end; i++) {
            double angle = vecAngles[i];
            // Cache index matches angle index since both grids share the same
            // step/range construction. If sizes diverge (defensive), snap.
            int cacheIdx = (i < (int)topCacheEntries.size()) ? i
                : findNearestCachedAngle(topLevel, angle);
            if (cacheIdx < 0)
                continue;
            const RotatedCacheEntry& entry = topCacheEntries[cacheIdx];
            const cv::Mat& rotT = entry.templ;
            if (rotT.empty() || srcTop.cols < rotT.cols || srcTop.rows < rotT.rows)
                continue;

            cv::Mat matResult;
            matchTemplateMatchTool8u(
                srcTop,
                rotT,
                entry.mean,
                entry.norm,
                entry.invArea,
                entry.resultEqual1,
                matResult,
                srcTopSum,
                srcTopSqsum);

            if (matResult.empty())
                continue;

            const float halfDw = (rotT.cols - templTop.cols) / 2.0f;
            const float halfDh = (rotT.rows - templTop.rows) / 2.0f;
            const double angleRad = angle * D2R;

            for (int j = 0; j < maxTargets + MATCH_CANDIDATE_NUM - 1; j++) {
                double dMaxVal;
                cv::Point ptMaxLoc;
                cv::minMaxLoc(matResult, nullptr, &dMaxVal, nullptr, &ptMaxLoc);

                if (dMaxVal < vecLayerScore[topLevel])
                    break;

                // ptMaxLoc is the top-left of the rotated template bbox in
                // un-rotated source coords. The "un-rotated template" TL in
                // source coords is offset by half the bbox-vs-template diff.
                cv::Point2f ptLT_orig(
                    ptMaxLoc.x + halfDw,
                    ptMaxLoc.y + halfDh);

                MatchCandidate mc;
                // Stored in pre-rotated form so the existing refinement code
                // (ptRotatePt2f(cand.pt, ptCenter, -cand.angle*D2R)) recovers ptLT_orig.
                mc.pt = ptRotatePt2f(ptLT_orig, ptCenter, angleRad);
                mc.score = dMaxVal;
                mc.angle = angle;
                localMatches.push_back(mc);

                GetNextMaxLoc(
                    matResult,
                    ptMaxLoc,
                    templTop.size(),
                    dMaxVal,
                    m_maxOverlap_topLevel);
            }
        }

        if (!localMatches.empty()) {
            std::lock_guard<std::mutex> lk(topLevelMtx);
            vecMatchParameter.insert(vecMatchParameter.end(), localMatches.begin(), localMatches.end());
        }
    });

    if (vecMatchParameter.empty())
        return {};

    std::sort(vecMatchParameter.begin(), vecMatchParameter.end(),
        [](const MatchCandidate& a, const MatchCandidate& b) { return a.score > b.score; });

    std::vector<MatchCandidate> vecFilteredCandidates;
    float distThreshold = std::min(m_templatePyramid[topLevel].rows / 2, m_templatePyramid[topLevel].cols / 2) *
        (1.f - m_maxOverlap_topLevel);

    for (const auto& candidate : vecMatchParameter) {
        bool bOverlap = false;
        for (const auto& accepted : vecFilteredCandidates) {
            float dx = candidate.pt.x - accepted.pt.x;
            float dy = candidate.pt.y - accepted.pt.y;
            if (std::sqrt(dx * dx + dy * dy) < distThreshold) {
                bOverlap = true;
                break;
            }
        }
        if (!bOverlap)
            vecFilteredCandidates.push_back(candidate);
    }

    size_t numToRefine = vecFilteredCandidates.size();
    std::vector<MatchInfo> vecAllResult;
    std::mutex refineMtx;
    // Refine down to minPyramidLevel (0 = full resolution). Clamp to valid range.
    int iStopLayer = std::max(0, std::min(minPyramidLevel, topLevel));
    int iW = m_templatePyramid[0].cols;
    int iH = m_templatePyramid[0].rows;

    // Precompute integrals of each pyramid level once per Match() call. With
    // template-rotation refinement, the source layer is constant across all
    // candidates/angles, so a single integral per level serves every call.
    const int numSrcLevels = static_cast<int>(srcPyramid.size());
    std::vector<cv::Mat> srcLayerSums(numSrcLevels);
    std::vector<cv::Mat> srcLayerSqsums(numSrcLevels);
    for (int lv = iStopLayer; lv <= topLevel - 1 && lv < numSrcLevels; lv++) {
        cv::integral(srcPyramid[lv], srcLayerSums[lv], srcLayerSqsums[lv], CV_64F);
    }

    cv::parallel_for_(cv::Range(0, (int)numToRefine), [&](const cv::Range& range) {
        std::vector<MatchInfo> localResults;
        localResults.reserve((size_t)(range.end - range.start));

        for (int i = range.start; i < range.end; i++) {
            const MatchCandidate& cand = vecFilteredCandidates[i];

            double dRAngle = -cand.angle * D2R;
            cv::Point2f ptLT = ptRotatePt2f(cand.pt, ptCenter, dRAngle);

            double currentAngle = cand.angle;
            double currentScore = cand.score;
            int lastValidLayer = topLevel;

            for (int iLayer = topLevel - 1; iLayer >= iStopLayer; iLayer--) {
                const int iPadding = 6;
                const int iHalfPadding = iPadding / 2;

                ptLT *= 2.0f;

                std::array<double, 3> searchAngles{};
                int numAngles = 0;
                double layerAngleStep = 0.0;

                const cv::Mat& templLayer = m_templatePyramid[iLayer];
                if (only_search_angle) {
                    searchAngles[0] = currentAngle;
                    numAngles = 1;
                }
                else {
                    const int iMaxDim = std::max(templLayer.cols, templLayer.rows);
                    layerAngleStep = atan(2.0 / iMaxDim) * R2D;
                    searchAngles[0] = currentAngle - layerAngleStep;
                    searchAngles[1] = currentAngle;
                    searchAngles[2] = currentAngle + layerAngleStep;
                    numAngles = 3;
                }

                const cv::Mat& srcLayer = srcPyramid[iLayer];
                const cv::Mat& srcSum = srcLayerSums[iLayer];
                const cv::Mat& srcSqsum = srcLayerSqsums[iLayer];
                const cv::Rect srcRect(0, 0, srcLayer.cols, srcLayer.rows);
                if (iLayer >= (int)m_rotCache.size() || m_rotCache[iLayer].empty()) {
                    lastValidLayer = iLayer;
                    break;
                }
                const std::vector<RotatedCacheEntry>& cacheLayer = m_rotCache[iLayer];

                double bestScore = -1.0;
                cv::Point2f bestPt = ptLT;
                double bestAngle = currentAngle;

                // Per-angle correlation against an un-rotated source ROI using
                // the cached rotated template. Returns peak score, location in
                // matResult coords, and the geometry needed to map to source.
                auto correlateAngle = [&](double searchAngle,
                                          cv::Mat& matResult,
                                          double& outScore,
                                          cv::Point& outMaxLoc,
                                          cv::Point& outClippedTL,
                                          float& outHalfDw,
                                          float& outHalfDh) -> bool {
                    int cacheIdx = findNearestCachedAngle(iLayer, searchAngle);
                    if (cacheIdx < 0 || cacheIdx >= (int)cacheLayer.size())
                        return false;
                    const RotatedCacheEntry& entry = cacheLayer[cacheIdx];
                    if (entry.templ.empty())
                        return false;
                    const cv::Mat& rotT = entry.templ;
                    outHalfDw = (rotT.cols - templLayer.cols) / 2.0f;
                    outHalfDh = (rotT.rows - templLayer.rows) / 2.0f;

                    const cv::Point roiTL(
                        cvRound(ptLT.x - outHalfDw - iHalfPadding),
                        cvRound(ptLT.y - outHalfDh - iHalfPadding));
                    const cv::Size roiSize(rotT.cols + iPadding, rotT.rows + iPadding);
                    cv::Rect roiRect(roiTL, roiSize);
                    cv::Rect clipped = roiRect & srcRect;
                    if (clipped.width < rotT.cols || clipped.height < rotT.rows)
                        return false;

                    cv::Mat matROI = srcLayer(clipped);
                    simdCCORR_8u(matROI, rotT, matResult);
                    CCOEFF_Denominator_8u(
                        matROI, rotT,
                        entry.mean, entry.norm, entry.invArea, entry.resultEqual1,
                        matResult,
                        srcSum, srcSqsum,
                        cv::Point(clipped.x, clipped.y));
                    if (matResult.empty())
                        return false;

                    cv::minMaxLoc(matResult, nullptr, &outScore, nullptr, &outMaxLoc);
                    outClippedTL = clipped.tl();
                    return true;
                };

                const bool trySubPixelHere =
                    (iLayer == 0 && m_useSubPixel && numAngles == 3 && !only_search_angle);

                if (trySubPixelHere) {
                    float vecScore[3][3][3];
                    cv::Point clippedTL[3];
                    cv::Point maxLoc[3];
                    cv::Mat results[3];
                    float halfDw_k[3] = { 0.f, 0.f, 0.f };
                    float halfDh_k[3] = { 0.f, 0.f, 0.f };
                    double scoreArr[3] = { 0, 0, 0 };
                    bool border[3] = { false, false, false };
                    bool hypOk[3] = { false, false, false };

                    for (int k = 0; k < 3; k++) {
                        cv::Mat mr;
                        double sc;
                        cv::Point ml, ctl;
                        float hdw, hdh;
                        if (!correlateAngle(searchAngles[k], mr, sc, ml, ctl, hdw, hdh))
                            continue;
                        hypOk[k] = true;
                        scoreArr[k] = sc;
                        maxLoc[k] = ml;
                        clippedTL[k] = ctl;
                        halfDw_k[k] = hdw;
                        halfDh_k[k] = hdh;
                        results[k] = mr;
                        border[k] = (ml.x == 0 || ml.y == 0 ||
                            ml.x == mr.cols - 1 || ml.y == mr.rows - 1);
                        for (int yy = -1; yy <= 1; yy++) {
                            for (int xx = -1; xx <= 1; xx++) {
                                vecScore[k][xx + 1][yy + 1] = mr.at<float>(ml + cv::Point(xx, yy));
                            }
                        }
                    }

                    if (hypOk[0] && hypOk[1] && hypOk[2]) {
                        int iMax = 0;
                        for (int k = 1; k < 3; k++) {
                            if (scoreArr[k] > scoreArr[iMax])
                                iMax = k;
                        }

                        bestScore = scoreArr[iMax];

                        if (bestScore < vecLayerScore[iLayer]) {
                            lastValidLayer = iLayer;
                            break;
                        }

                        if (iMax == 1 && !border[1]) {
                            double nx = 0, ny = 0, nAngDeg = 0;
                            if (matchToolSubPixelEstimation(
                                    vecScore, 1, layerAngleStep,
                                    static_cast<double>(maxLoc[1].x),
                                    static_cast<double>(maxLoc[1].y),
                                    searchAngles[1],
                                    &nx, &ny, &nAngDeg)) {
                                bestPt = cv::Point2f(
                                    clippedTL[1].x + static_cast<float>(nx) + halfDw_k[1],
                                    clippedTL[1].y + static_cast<float>(ny) + halfDh_k[1]);
                                bestAngle = nAngDeg;
                            }
                            else {
                                bestPt = cv::Point2f(
                                    clippedTL[iMax].x + static_cast<float>(maxLoc[iMax].x) + halfDw_k[iMax],
                                    clippedTL[iMax].y + static_cast<float>(maxLoc[iMax].y) + halfDh_k[iMax]);
                                bestAngle = searchAngles[iMax];
                            }
                        }
                        else {
                            bestPt = cv::Point2f(
                                clippedTL[iMax].x + static_cast<float>(maxLoc[iMax].x) + halfDw_k[iMax],
                                clippedTL[iMax].y + static_cast<float>(maxLoc[iMax].y) + halfDh_k[iMax]);
                            bestAngle = searchAngles[iMax];
                        }
                    }
                    else {
                        // Some hypothesis failed — fall back to plain max search.
                        for (int k = 0; k < numAngles; k++) {
                            cv::Mat mr;
                            double sc;
                            cv::Point ml, ctl;
                            float hdw, hdh;
                            if (!correlateAngle(searchAngles[k], mr, sc, ml, ctl, hdw, hdh))
                                continue;
                            if (sc > bestScore) {
                                bestScore = sc;
                                bestAngle = searchAngles[k];
                                bestPt = cv::Point2f(
                                    ctl.x + static_cast<float>(ml.x) + hdw,
                                    ctl.y + static_cast<float>(ml.y) + hdh);
                            }
                        }
                    }
                }
                else {
                    for (int k = 0; k < numAngles; k++) {
                        cv::Mat mr;
                        double sc;
                        cv::Point ml, ctl;
                        float hdw, hdh;
                        if (!correlateAngle(searchAngles[k], mr, sc, ml, ctl, hdw, hdh))
                            continue;
                        if (sc > bestScore) {
                            bestScore = sc;
                            bestAngle = searchAngles[k];
                            bestPt = cv::Point2f(
                                ctl.x + static_cast<float>(ml.x) + hdw,
                                ctl.y + static_cast<float>(ml.y) + hdh);
                        }
                    }
                }

                ptLT = bestPt;
                currentAngle = bestAngle;
                currentScore = bestScore;

                if (currentScore < vecLayerScore[iLayer]) {
                    lastValidLayer = iLayer;
                    break;
                }
                lastValidLayer = iLayer;
            }

            if (currentScore >= m_scoreThreshold) {
                int scaleFactor = (lastValidLayer == 0) ? 1 : (1 << lastValidLayer);
                cv::Point2f ptLTScaled = ptLT * (float)scaleFactor;

                double dRAngleBase = -currentAngle * D2R;

                cv::Point2f ptRT(
                    ptLTScaled.x + static_cast<float>(iW * cos(dRAngleBase)),
                    ptLTScaled.y - static_cast<float>(iW * sin(dRAngleBase)));
                cv::Point2f ptLB(
                    ptLTScaled.x + static_cast<float>(iH * sin(dRAngleBase)),
                    ptLTScaled.y + static_cast<float>(iH * cos(dRAngleBase)));
                cv::Point2f ptRB(
                    ptRT.x + static_cast<float>(iH * sin(dRAngleBase)),
                    ptRT.y + static_cast<float>(iH * cos(dRAngleBase)));

                cv::Point2d ptCenterCalc(
                    (ptLTScaled.x + ptRT.x + ptRB.x + ptLB.x) / 4.0,
                    (ptLTScaled.y + ptRT.y + ptRB.y + ptLB.y) / 4.0);

                localResults.emplace_back(currentScore, currentAngle, ptCenterCalc);
            }
        }

        if (!localResults.empty()) {
            std::lock_guard<std::mutex> lk(refineMtx);
            vecAllResult.insert(vecAllResult.end(), localResults.begin(), localResults.end());
        }
    });

    if (vecAllResult.empty())
        return {};

    return filterMatches(vecAllResult, maxTargets);
}

std::vector<MatchInfo> MatchToolNCCMatcherImpl::filterMatches(std::vector<MatchInfo>& matches, int maxTargets) {
    if (matches.empty())
        return matches;

    std::sort(matches.begin(), matches.end(),
        [](const MatchInfo& a, const MatchInfo& b) { return a.score > b.score; });

    const int templW = m_templateGray8u.cols;
    const int templH = m_templateGray8u.rows;
    const double templArea = static_cast<double>(templW) * templH;

    std::vector<bool> deleted(matches.size(), false);

    for (size_t i = 0; i < matches.size(); ++i) {
        if (deleted[i])
            continue;

        cv::RotatedRect rect1(
            matches[i].position,
            cv::Size2f((float)templW, (float)templH),
            (float)matches[i].angle);

        for (size_t j = i + 1; j < matches.size(); ++j) {
            if (deleted[j])
                continue;

            cv::RotatedRect rect2(
                matches[j].position,
                cv::Size2f((float)templW, (float)templH),
                (float)matches[j].angle);

            std::vector<cv::Point2f> interPts;
            int interType = cv::rotatedRectangleIntersection(rect1, rect2, interPts);

            if (interType == cv::INTERSECT_NONE)
                continue;
            else if (interType == cv::INTERSECT_FULL) {
                size_t delIdx = (matches[i].score >= matches[j].score) ? j : i;
                deleted[delIdx] = true;
                if (delIdx == i)
                    break;
            }
            else {
                if (interPts.size() < 3)
                    continue;

                cv::Point2f center(0.f, 0.f);
                for (const auto& p : interPts)
                    center += p;
                center *= (1.f / interPts.size());

                std::sort(interPts.begin(), interPts.end(),
                    [&center](const cv::Point2f& a, const cv::Point2f& b) {
                        return atan2(a.y - center.y, a.x - center.x) <
                            atan2(b.y - center.y, b.x - center.x);
                    });

                double interArea = std::fabs(cv::contourArea(interPts));
                double overlapRatio = interArea / templArea;

                if (overlapRatio > m_maxOverlap) {
                    size_t delIdx = (matches[i].score >= matches[j].score) ? j : i;
                    deleted[delIdx] = true;
                    if (delIdx == i)
                        break;
                }
            }
        }
    }

    std::vector<MatchInfo> result;
    result.reserve(maxTargets);
    for (size_t i = 0; i < matches.size(); ++i) {
        if (!deleted[i]) {
            result.push_back(matches[i]);
            if ((int)result.size() >= maxTargets)
                break;
        }
    }
    return result;
}

void MatchToolNCCMatcherImpl::setAngleRange(double minA, double maxA) {
    m_minAngle = minA;
    m_maxAngle = maxA;
}

void MatchToolNCCMatcherImpl::setScoreThreshold(double t) {
    m_scoreThreshold = std::max(0.0, std::min(1.0, t));
}

void MatchToolNCCMatcherImpl::setMaxOverlap(double o) {
    m_maxOverlap = std::max(0.0, std::min(1.0, o));
}

void MatchToolNCCMatcherImpl::setUseSubPixel(bool enable) {
    m_useSubPixel = enable;
}

bool MatchToolNCCMatcherImpl::getUseSubPixel() const {
    return m_useSubPixel;
}

double MatchToolNCCMatcherImpl::getMinAngle() const { return m_minAngle; }
double MatchToolNCCMatcherImpl::getMaxAngle() const { return m_maxAngle; }
double MatchToolNCCMatcherImpl::getScoreThreshold() const { return m_scoreThreshold; }
double MatchToolNCCMatcherImpl::getMaxOverlap() const { return m_maxOverlap; }

std::vector<cv::Mat> MatchToolNCCMatcherImpl::getTemplatePyramid() const { return m_templatePyramid; }
std::vector<cv::Scalar> MatchToolNCCMatcherImpl::getVecTemplMean() const { return m_vecTemplMean; }
std::vector<double> MatchToolNCCMatcherImpl::getVecTemplNorm() const { return m_vecTemplNorm; }
std::vector<double> MatchToolNCCMatcherImpl::getVecTemplInvArea() const { return m_vecInvArea; }
std::vector<bool> MatchToolNCCMatcherImpl::getVecResultEqual1() const { return m_vecResultEqual1; }
bool MatchToolNCCMatcherImpl::getIsLearned() const { return m_isLearned; }
int MatchToolNCCMatcherImpl::getNumLevels() const { return num_levels; }
int MatchToolNCCMatcherImpl::getNumLevelsForSrc() const { return num_levels_for_src; }

void MatchToolNCCMatcherImpl::setNumLevels(int n) { num_levels = n; }
void MatchToolNCCMatcherImpl::setNumLevelsForSrc(int n) { num_levels_for_src = n; }

void MatchToolNCCMatcherImpl::setTemplatePyramid(const std::vector<cv::Mat>& p) { m_templatePyramid = p; }
void MatchToolNCCMatcherImpl::setVecTemplMean(const std::vector<cv::Scalar>& v) { m_vecTemplMean = v; }
void MatchToolNCCMatcherImpl::setVecTemplNorm(const std::vector<double>& v) { m_vecTemplNorm = v; }
void MatchToolNCCMatcherImpl::setVecTemplInvArea(const std::vector<double>& v) { m_vecInvArea = v; }
void MatchToolNCCMatcherImpl::setVecResultEqual1(const std::vector<bool>& v) { m_vecResultEqual1 = v; }

} // namespace pattern_matching
} // namespace cv
