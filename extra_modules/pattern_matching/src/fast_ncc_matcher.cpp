// ================================================================================================
// fast_ncc_matcher.cpp - Improved Implementation (MatchTool approach)
// ================================================================================================
#include "fast_ncc_matcher.h"
#include "opencvsharp_license_guard.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/imgproc.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif

#define D2R (CV_PI / 180.0)
#define R2D (180.0 / CV_PI)
#define VISION_TOLERANCE 0.0000001
#define MATCH_CANDIDATE_NUM 5


namespace cv {
    namespace pattern_matching {
        static void CCOEFF_Denominator(
            const cv::Mat& matSrc,      // CV_32FC4
            const cv::Mat& templ,
            const cv::Scalar& templMean,
            double templNorm,
            double invArea,
            bool resultEqual1,
            cv::Mat& matResult
        )
        {
            if (resultEqual1 || templNorm < DBL_EPSILON) {
                matResult.setTo(1.0f);
                return;
            }

            const int th = templ.rows;
            const int tw = templ.cols;

            // ============================================================
            // 1. Compute integral images (Vec4d)
            // ============================================================
            cv::Mat sum, sqsum;
            cv::integral(matSrc, sum, sqsum, CV_64F); // sum & sqsum are CV_64FC4

            const cv::Vec4d tMean(
                templMean[0], templMean[1],
                templMean[2], templMean[3]
            );

            // ============================================================
            // 2. Parallel NCC denominator computation
            // ============================================================
            //#pragma omp parallel for schedule(static)
#pragma omp parallel for schedule(static) if(matResult.rows > 32)
            for (int i = 0; i < matResult.rows; i++)
            {
                float* rrow = matResult.ptr<float>(i);

                const cv::Vec4d* s0 = sum.ptr<cv::Vec4d>(i);
                const cv::Vec4d* s1 = sum.ptr<cv::Vec4d>(i + th);
                const cv::Vec4d* q0 = sqsum.ptr<cv::Vec4d>(i);
                const cv::Vec4d* q1 = sqsum.ptr<cv::Vec4d>(i + th);

                for (int j = 0; j < matResult.cols; j++)
                {
                    double num = rrow[j]; // CCOR numerator

                    // Integral window fetch (SIMD-friendly)
                    cv::Vec4d wndSum =
                        s0[j] - s0[j + tw] - s1[j] + s1[j + tw];

                    cv::Vec4d wndSqSum =
                        q0[j] - q0[j + tw] - q1[j] + q1[j + tw];

                    // Vector math
                    double sumSq =
                        wndSum.dot(wndSum);

                    double sqSum =
                        wndSqSum[0] + wndSqSum[1] +
                        wndSqSum[2] + wndSqSum[3];

                    num -= wndSum.dot(tMean);

                    double wndMean2 = sumSq * invArea;
                    double diff2 = std::max(sqSum - wndMean2, 0.0);

                    double denom;
                    if (diff2 <= std::min(0.5, 10 * FLT_EPSILON * sqSum))
                        denom = 0.0;
                    else
                        denom = std::sqrt(diff2) * templNorm;

                    // NCC clamp
                    if (fabs(num) < denom)
                        num /= denom;
                    else if (fabs(num) < denom * 1.125)
                        num = num > 0 ? 1.0 : -1.0;
                    else
                        num = 0.0;

                    rrow[j] = (float)num;
                }
            }
        }



        // Helper: Rotate point around center
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

        // Helper: Get best rotation size
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

            float fH1 = sizeDst.width * sin(dAngle * D2R) * cos(dAngle * D2R);
            float fH2 = sizeDst.height * sin(dAngle * D2R) * cos(dAngle * D2R);

            int iHalfHeight = (int)ceil(fTopY - ptCenter.y - fH1);
            int iHalfWidth = (int)ceil(fRightX - ptCenter.x - fH2);

            cv::Size sizeRet(iHalfWidth * 2, iHalfHeight * 2);

            bool bWrongSize = (sizeDst.width < sizeRet.width && sizeDst.height > sizeRet.height) ||
                (sizeDst.width > sizeRet.width && sizeDst.height < sizeRet.height) ||
                (sizeDst.area() > sizeRet.area());

            if (bWrongSize)
                sizeRet = cv::Size(int(fRightX - fLeftX + 0.5), int(fTopY - fBottomY + 0.5));

            return sizeRet;
        }


        static cv::Point GetNextMaxLoc(
            cv::Mat& matResult,
            const cv::Point& ptMaxLoc,
            const cv::Size& sizeTemplate,
            double& dMaxValue,
            double dMaxOverlap)
        {
            int iStartX = ptMaxLoc.x - sizeTemplate.width * (1 - dMaxOverlap);
            int iStartY = ptMaxLoc.y - sizeTemplate.height * (1 - dMaxOverlap);

            cv::rectangle(matResult, Rect(iStartX, iStartY, 2 * sizeTemplate.width * (1 - dMaxOverlap), 2 * sizeTemplate.height * (1 - dMaxOverlap)), Scalar(-1), cv::FILLED);

            cv::Point ptNew;
            cv::minMaxLoc(matResult, nullptr, &dMaxValue, nullptr, &ptNew);
            return ptNew;
        }


        cv::Ptr<FastNCCMatcher> FastNCCMatcher::create() {
            return cv::makePtr<FastNCCMatcherImpl>();
        }

        std::vector<MatchInfo> FastNCCMatcher::MatchTemplate(
            cv::InputArray source,
            cv::InputArray templateImg,
            double scoreThreshold,
            double angleRange,
            double maxOverlap,
            int minReducedArea,
            int maxTargets) {

            cv::Ptr<FastNCCMatcher> matcher = FastNCCMatcher::create();

            if (!matcher->setTemplate(templateImg)) {
                return std::vector<MatchInfo>();
            }

            // Use a reasonable default number of pyramid levels for the convenience API.
            // (Callers that need full control should use the instance API.)
            const int defaultNumLevels = 8;
            if (!matcher->train(defaultNumLevels, minReducedArea)) {
                return std::vector<MatchInfo>();
            }

            matcher->setAngleRange(-angleRange, angleRange);
            matcher->setScoreThreshold(scoreThreshold);
            matcher->setMaxOverlap(maxOverlap);

            //return matcher->match(source, maxTargets, minReducedArea);

            return matcher->match(source, defaultNumLevels, maxTargets, 0);
        }

        FastNCCMatcherImpl::FastNCCMatcherImpl()
            : m_isTemplateSet(false)
            , m_minAngle(-180.0)
            , m_maxAngle(180.0)
            , m_scoreThreshold(0.7)
            , m_maxOverlap(0.5)
            , m_borderColor(0)
            , num_levels(0)
            , num_levels_for_src(0)
            , m_isLearned(false) {
        }

        FastNCCMatcherImpl::~FastNCCMatcherImpl() {}


        bool FastNCCMatcherImpl::setTemplate(cv::InputArray templateImg)
        {
            cv::Mat temp = templateImg.getMat();
            if (temp.empty()) return false;

            if (temp.channels() == 3)
                cv::cvtColor(temp, m_templateImage, cv::COLOR_BGR2BGRA);
            else if (temp.channels() == 1)
                cv::cvtColor(temp, m_templateImage, cv::COLOR_GRAY2BGRA);
            else
                m_templateImage = temp.clone();

            m_templateImage.convertTo(m_templateImage, CV_32FC4);

            // MatchTool logic
            double meanVal = cv::mean(m_templateImage)[0];
            m_borderColor = meanVal < 128 ? 255 : 0;

            m_isTemplateSet = true;
            m_templatePyramid.clear();
            m_isLearned = false;
            return true;
        }

        bool FastNCCMatcherImpl::train(int numLevels, int minReducedArea = 256)
        {

            if (!m_isTemplateSet) return false;


            buildPyramid(m_templateImage, m_templatePyramid, numLevels, minReducedArea);


            int n = (int)m_templatePyramid.size();

            // Allocate stats
            int L = (int)m_templatePyramid.size();
            m_vecTemplMean.resize(L);
            m_vecTemplNorm.resize(L);
            m_vecInvArea.resize(L);
            m_vecResultEqual1.resize(L, false);

            for (int i = 0; i < n; i++)
            {
                const cv::Mat& T = m_templatePyramid[i];

                double invArea = 1.0 / (T.rows * T.cols);
                cv::Scalar mean, stddev;

                cv::meanStdDev(T, mean, stddev);

                // Sum of variance across channels
                double templNorm =
                    stddev[0] * stddev[0] +
                    stddev[1] * stddev[1] +
                    stddev[2] * stddev[2] +
                    stddev[3] * stddev[3];

                if (templNorm < DBL_EPSILON)
                {
                    m_vecResultEqual1[i] = true;
                }

                templNorm = std::sqrt(templNorm);
                templNorm /= std::sqrt(invArea);  // scale invariant

                m_vecInvArea[i] = invArea;
                m_vecTemplMean[i] = mean;
                m_vecTemplNorm[i] = templNorm;
            }

            m_isLearned = true;
            return true;
        }

        void FastNCCMatcherImpl::buildPyramid(const cv::Mat& image, std::vector<cv::Mat>& pyramid, int numLevels, int minArea) {
            pyramid.clear();
            if (image.empty()) return;

            cv::Mat current;
            if (image.type() != CV_32FC4) {
                image.convertTo(current, CV_32FC4);
            }
            else {
                current = image.clone();
            }

            pyramid.push_back(current);

            int currentArea = current.rows * current.cols;

            num_levels = 0;

            for (int i = 0; i < numLevels; i++)
            {

                if (currentArea <= minArea) {

                    break;
                }

                cv::Mat down;
                cv::pyrDown(current, down);
                if (down.empty() || down.rows < 4 || down.cols < 4) break;

                if (down.type() != CV_32FC4) {
                    down.convertTo(down, CV_32FC4);
                }

                pyramid.push_back(down);
                current = down;
                currentArea = current.rows * current.cols;
            }

            num_levels = pyramid.size();

            /*while (currentArea > minArea) {
                cv::Mat down;
                cv::pyrDown(current, down);
                if (down.empty() || down.rows < 4 || down.cols < 4) break;

                if (down.type() != CV_32FC4) {
                    down.convertTo(down, CV_32FC4);
                }

                pyramid.push_back(down);
                current = down;
                currentArea = current.rows * current.cols;
            }*/
        }

        void FastNCCMatcherImpl::buildSrcPyramid(const cv::Mat& image, std::vector<cv::Mat>& pyramid, int topLayer) {
            pyramid.clear();
            if (image.empty()) return;

            cv::Mat current;

            if (image.type() != CV_32FC4) {
                image.convertTo(current, CV_32FC4);
            }
            else {
                current = image.clone();
            }

            pyramid.push_back(current);
            for (int i = 0; i < topLayer; i++)
            {
                cv::Mat down;
                cv::pyrDown(current, down);

                if (down.type() != CV_32FC4) {
                    down.convertTo(down, CV_32FC4);
                }

                pyramid.push_back(down);
                current = down;

            }
        }

        std::vector<MatchInfo> FastNCCMatcherImpl::match(
            cv::InputArray source,
            int numLevels,
            int maxTargets,
            int minPyramidLevel) {

            if (!opencvsharp_license_runtime_activated())
                return std::vector<MatchInfo>();

            cv::Mat src = source.getMat();
            if (src.empty() || m_templatePyramid.empty()) {
                return std::vector<MatchInfo>();
            }

            if ((m_templateImage.cols < src.cols && m_templateImage.rows > src.rows) || (m_templateImage.cols > src.cols && m_templateImage.rows < src.rows)) {
                return std::vector<MatchInfo>();
            }

            cv::Mat srcFloat4;

            if (src.channels() == 1)
            {
                cv::cvtColor(src, srcFloat4, cv::COLOR_GRAY2BGRA);
            }
            else if (src.channels() == 3)
            {
                cv::cvtColor(src, srcFloat4, cv::COLOR_BGR2BGRA);
            }
            else if (src.channels() == 4)
            {
                srcFloat4 = src.clone();
            }

            srcFloat4.convertTo(srcFloat4, CV_32FC4);


            // Build source pyramid
            std::vector<cv::Mat> srcPyramid;
            buildSrcPyramid(srcFloat4, srcPyramid, static_cast<int>(m_templatePyramid.size()) - 1);


            if (srcPyramid.empty()) return std::vector<MatchInfo>();

            int topLevel = std::min(
                static_cast<int>(m_templatePyramid.size()),
                static_cast<int>(srcPyramid.size())
            ) - 1;

            topLevel = std::min(topLevel, numLevels);

            num_levels_for_src = topLevel;

            //std::cout << "Top level: " << topLevel << std::endl;

            if (topLevel < 0) return std::vector<MatchInfo>();

            // IMPROVEMENT 1: Calculate angle step like MatchTool (based on template diagonal)
            int iMaxDemision = std::max(m_templatePyramid[topLevel].cols,
                m_templatePyramid[topLevel].rows);
            double dAngleStep = atan(2.0 / iMaxDemision) * R2D;

            // Generate angle list
            std::vector<double> vecAngles;
            bool only_search_angle = false;
            if (std::abs(m_maxAngle - m_minAngle) < VISION_TOLERANCE) {
                vecAngles.push_back(m_minAngle);
                only_search_angle = true;
            }
            else {
                for (double dAngle = m_minAngle; dAngle <= m_maxAngle; dAngle += dAngleStep) {
                    vecAngles.push_back(dAngle);
                }
                // Ensure max angle is included
                if (vecAngles.empty() || std::abs(vecAngles.back() - m_maxAngle) > VISION_TOLERANCE) {
                    vecAngles.push_back(m_maxAngle);
                }
            }

            // IMPROVEMENT 2: Layer-based threshold reduction (like MatchTool)
            // Each pyramid level uses 90% of previous threshold for more permissive coarse search
            int iTopLayer = topLevel;
            std::vector<double> vecLayerScore(iTopLayer + 1);
            vecLayerScore[0] = m_scoreThreshold;
            for (int i = 1; i <= iTopLayer; i++) {
                vecLayerScore[i] = vecLayerScore[i - 1] * 0.9;  // MatchTool uses 0.9 multiplier
            }

            // Phase 1: Coarse search at top level
            int iTopSrcW = srcPyramid[topLevel].cols;
            int iTopSrcH = srcPyramid[topLevel].rows;
            cv::Point2f ptCenter((iTopSrcW - 1) / 2.0f, (iTopSrcH - 1) / 2.0f);

            float m_maxOverlap_topLevel = m_maxOverlap;

            //float m_maxOverlap_topLevel = std::min(std::pow(1.2, topLevel) * m_maxOverlap, 0.75);

            if (0 < m_maxOverlap <= 0.3)
            {
                m_maxOverlap_topLevel = std::min(std::pow(1.2, topLevel) * m_maxOverlap, 1.0);
            }
            else if (m_maxOverlap == 0) {
                m_maxOverlap_topLevel = std::min(0.05 * topLevel, 1.0);
            }


            struct MatchCandidate {
                cv::Point2f pt;
                double score;
                double angle;
            };

            std::vector<MatchCandidate> vecMatchParameter;

#pragma omp parallel
            {
                std::vector<MatchCandidate> localMatches;
                localMatches.reserve(maxTargets + MATCH_CANDIDATE_NUM);

#pragma omp for schedule(dynamic, 1)
                for (int i = 0; i < (int)vecAngles.size(); i++) {
                    double angle = vecAngles[i];

                    cv::Size sizeBest = GetBestRotationSize(
                        srcPyramid[topLevel].size(),
                        m_templatePyramid[topLevel].size(),
                        angle);

                    float fTx = (sizeBest.width - 1) / 2.0f - ptCenter.x;
                    float fTy = (sizeBest.height - 1) / 2.0f - ptCenter.y;

                    cv::Mat matR = cv::getRotationMatrix2D(ptCenter, angle, 1.0);
                    matR.at<double>(0, 2) += fTx;
                    matR.at<double>(1, 2) += fTy;

                    cv::Mat matRotated;
                    cv::warpAffine(
                        srcPyramid[topLevel],
                        matRotated,
                        matR,
                        sizeBest,
                        cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT,
                        cv::Scalar(m_borderColor));

                    // *** ADD THIS CHECK BEFORE MATCHTEMPLATE ***
                    if (matRotated.empty() ||
                        matRotated.cols < m_templatePyramid[topLevel].cols ||
                        matRotated.rows < m_templatePyramid[topLevel].rows) {
                        continue;  // Skip this angle
                    }

                    cv::Mat matResult;
                    cv::matchTemplate(
                        matRotated,
                        m_templatePyramid[topLevel],
                        matResult,
                        cv::TM_CCORR);

                    CCOEFF_Denominator(
                        matRotated,
                        m_templatePyramid[topLevel],
                        m_vecTemplMean[topLevel],
                        m_vecTemplNorm[topLevel],
                        m_vecInvArea[topLevel],
                        m_vecResultEqual1[topLevel],
                        matResult);

                    if (matResult.empty()) continue;

                    for (int j = 0; j < maxTargets + MATCH_CANDIDATE_NUM - 1; j++) {
                        double dMaxVal;
                        cv::Point ptMaxLoc;
                        cv::minMaxLoc(matResult, nullptr, &dMaxVal, nullptr, &ptMaxLoc);

                        if (dMaxVal < vecLayerScore[topLevel])
                            break;

                        MatchCandidate mc;
                        mc.pt = cv::Point2f(
                            ptMaxLoc.x - fTx,
                            ptMaxLoc.y - fTy);
                        mc.score = dMaxVal;
                        mc.angle = angle;

                        localMatches.push_back(mc);

                        GetNextMaxLoc(
                            matResult,
                            ptMaxLoc,
                            m_templatePyramid[topLevel].size(),
                            dMaxVal,
                            //m_maxOverlap
                            m_maxOverlap_topLevel
                        );
                    }
                }

#pragma omp critical
                vecMatchParameter.insert(
                    vecMatchParameter.end(),
                    localMatches.begin(),
                    localMatches.end());
            }

            if (vecMatchParameter.empty()) return {};

            // Sort by score descending
            std::sort(vecMatchParameter.begin(), vecMatchParameter.end(),
                [](const MatchCandidate& a, const MatchCandidate& b) {
                    return a.score > b.score;
                });

            std::vector<MatchCandidate> vecFilteredCandidates;


            /*float distThreshold = std::min(m_templatePyramid[topLevel].rows / 2, m_templatePyramid[topLevel].cols / 2) * (1. - m_maxOverlap);*/

            float distThreshold = std::min(m_templatePyramid[topLevel].rows / 2, m_templatePyramid[topLevel].cols / 2) * (1. - m_maxOverlap_topLevel);

            for (const auto& candidate : vecMatchParameter) {
                bool bOverlap = false;

                for (const auto& accepted : vecFilteredCandidates) {
                    float dx = candidate.pt.x - accepted.pt.x;
                    float dy = candidate.pt.y - accepted.pt.y;
                    float distance = std::sqrt(dx * dx + dy * dy);

                    if (distance < distThreshold) {
                        bOverlap = true;
                        break;
                    }
                }

                if (!bOverlap) {
                    vecFilteredCandidates.push_back(candidate);
                }
            }

            // IMPROVEMENT 4: Limit refinement candidates (MatchTool limits to best N)
            // MatchTool refines only top (maxTargets * 2) candidates
            /*size_t numToRefine = std::min(vecFilteredCandidates.size(),
                static_cast<size_t>(maxTargets * 5));*/

            size_t numToRefine = vecFilteredCandidates.size();

            //size_t numToRefine = std::max(static_cast<size_t>(maxTargets), static_cast<size_t>(vecFilteredCandidates.size() * 0.1));

            // ================= Phase 2: Refinement through pyramid (OpenMP optimized) =================
            std::vector<MatchInfo> vecAllResult;
            // Refine down to minPyramidLevel (0 = full resolution). Clamp to valid range.
            int iStopLayer = std::max(0, std::min(minPyramidLevel, topLevel));

            int iW = m_templatePyramid[0].cols;
            int iH = m_templatePyramid[0].rows;

#pragma omp parallel
            {
                std::vector<MatchInfo> localResults;
                localResults.reserve(8);

#pragma omp for schedule(dynamic, 1)
                for (int i = 0; i < (int)numToRefine; i++)
                {
                    const MatchCandidate& cand = vecFilteredCandidates[i];

                    double dRAngle = -cand.angle * D2R;
                    cv::Point2f ptLT = ptRotatePt2f(cand.pt, ptCenter, dRAngle);

                    double currentAngle = cand.angle;
                    double currentScore = cand.score;

                    int lastValidLayer = topLevel;

                    // ================= Refinement through pyramid =================
                    for (int iLayer = topLevel - 1; iLayer >= iStopLayer; iLayer--)
                    {
                        //int iPadding = 6 + (2 * (topLevel - iLayer));
                        int iPadding = 6;
                        int iHalfPadding = iPadding / 2;

                        ptLT *= 2.0f;

                        std::array<double, 3> searchAngles;
                        int numAngles = 0;

                        if (only_search_angle)
                        {
                            searchAngles[0] = currentAngle;
                            numAngles = 1;
                        }
                        else {
                            int iMaxDim = std::max(
                                m_templatePyramid[iLayer].cols,
                                m_templatePyramid[iLayer].rows
                            );
                            double layerAngleStep = atan(2.0 / iMaxDim) * R2D;

                            searchAngles[0] = currentAngle - layerAngleStep;
                            searchAngles[1] = currentAngle;
                            searchAngles[2] = currentAngle + layerAngleStep;
                            numAngles = 3;
                        }

                        cv::Point2f ptSrcCenter(
                            (srcPyramid[iLayer].cols - 1) / 2.0f,
                            (srcPyramid[iLayer].rows - 1) / 2.0f
                        );

                        double bestScore = -1.0;
                        cv::Point2f bestPt = ptLT;
                        double bestAngle = currentAngle;


                        for (int k = 0; k < numAngles; k++)
                        {
                            double searchAngle = searchAngles[k];
                            double rad = searchAngle * D2R;

                            cv::Point2f ptLT_rotate =
                                ptRotatePt2f(ptLT, ptSrcCenter, rad);

                            cv::Size sizePadding(
                                m_templatePyramid[iLayer].cols + iPadding,
                                m_templatePyramid[iLayer].rows + iPadding
                            );

                            cv::Mat rMat =
                                cv::getRotationMatrix2D(ptSrcCenter, searchAngle, 1.0);

                            rMat.at<double>(0, 2) -= (ptLT_rotate.x - iHalfPadding);
                            rMat.at<double>(1, 2) -= (ptLT_rotate.y - iHalfPadding);

                            cv::Mat matROI;
                            cv::warpAffine(
                                srcPyramid[iLayer],
                                matROI,
                                rMat,
                                sizePadding
                            );

                            if (matROI.cols < m_templatePyramid[iLayer].cols ||
                                matROI.rows < m_templatePyramid[iLayer].rows) {
                                continue;  // Skip this angle, ROI too small
                            }


                            cv::Mat matResult;
                            cv::matchTemplate(
                                matROI,
                                m_templatePyramid[iLayer],
                                matResult,
                                cv::TM_CCORR
                            );

                            CCOEFF_Denominator(
                                matROI,
                                m_templatePyramid[iLayer],
                                m_vecTemplMean[iLayer],
                                m_vecTemplNorm[iLayer],
                                m_vecInvArea[iLayer],
                                m_vecResultEqual1[iLayer],
                                matResult
                            );

                            if (matResult.empty())
                                continue;

                            double dMaxValue;
                            cv::Point ptMaxLoc;
                            cv::minMaxLoc(matResult, nullptr, &dMaxValue, nullptr, &ptMaxLoc);

                            if (dMaxValue > bestScore)
                            {
                                bestScore = dMaxValue;
                                bestAngle = searchAngle;

                                cv::Point2f ptMatchInRotatedSrc(
                                    (ptLT_rotate.x - iHalfPadding) + ptMaxLoc.x,
                                    (ptLT_rotate.y - iHalfPadding) + ptMaxLoc.y
                                );

                                bestPt = ptRotatePt2f(
                                    ptMatchInRotatedSrc,
                                    ptSrcCenter,
                                    -rad
                                );
                            }
                        }

                        ptLT = bestPt;
                        currentAngle = bestAngle;
                        currentScore = bestScore;

                        if (currentScore < vecLayerScore[iLayer])
                        {
                            lastValidLayer = iLayer;
                            break;
                        }

                        lastValidLayer = iLayer;
                    }

                    // ================= Final result =================
                    if (currentScore >= m_scoreThreshold)
                    {
                        int scaleFactor = (lastValidLayer == 0) ? 1 : (1 << lastValidLayer);
                        cv::Point2f ptLTScaled = ptLT * (float)scaleFactor;

                        double dRAngleBase = -currentAngle * D2R;

                        cv::Point2f ptRT(
                            ptLTScaled.x + iW * cos(dRAngleBase),
                            ptLTScaled.y - iW * sin(dRAngleBase)
                        );
                        cv::Point2f ptLB(
                            ptLTScaled.x + iH * sin(dRAngleBase),
                            ptLTScaled.y + iH * cos(dRAngleBase)
                        );
                        cv::Point2f ptRB(
                            ptRT.x + iH * sin(dRAngleBase),
                            ptRT.y + iH * cos(dRAngleBase)
                        );

                        cv::Point2d ptCenterCalc(
                            (ptLTScaled.x + ptRT.x + ptRB.x + ptLB.x) / 4.0,
                            (ptLTScaled.y + ptRT.y + ptRB.y + ptLB.y) / 4.0
                        );

                        localResults.emplace_back(
                            currentScore,
                            currentAngle,
                            ptCenterCalc
                        );
                    }
                }

#pragma omp critical
                vecAllResult.insert(
                    vecAllResult.end(),
                    localResults.begin(),
                    localResults.end()
                );
            }


            if (vecAllResult.empty()) return std::vector<MatchInfo>();

            //std::cout << "vecAllResult Length: " << vecAllResult.size() << std::endl;

            // Filter overlapping matches
            std::vector<MatchInfo> filteredMatches = filterMatches(vecAllResult, maxTargets);
            return filteredMatches;
        }


        std::vector<MatchInfo> FastNCCMatcherImpl::filterMatches(
            std::vector<MatchInfo>& matches,
            int maxTargets)
        {
            if (matches.empty()) return matches;

            // Sort by score descending (CCOEFF_NORMED style)
            std::sort(matches.begin(), matches.end(),
                [](const MatchInfo& a, const MatchInfo& b) {
                    return a.score > b.score;
                });

            const int templW = m_templateImage.cols;
            const int templH = m_templateImage.rows;

            const double templArea = static_cast<double>(templW) * templH;

            std::vector<bool> deleted(matches.size(), false);

            for (size_t i = 0; i < matches.size(); ++i)
            {
                if (deleted[i]) continue;

                // Build rotated rect for match i
                cv::RotatedRect rect1(
                    matches[i].position,
                    cv::Size2f((float)templW, (float)templH),
                    (float)matches[i].angle
                );

                for (size_t j = i + 1; j < matches.size(); ++j)
                {
                    if (deleted[j]) continue;

                    cv::RotatedRect rect2(
                        matches[j].position,
                        cv::Size2f((float)templW, (float)templH),
                        (float)matches[j].angle
                    );

                    std::vector<cv::Point2f> interPts;
                    int interType = cv::rotatedRectangleIntersection(rect1, rect2, interPts);

                    if (interType == cv::INTERSECT_NONE)
                    {
                        continue;
                    }
                    else if (interType == cv::INTERSECT_FULL)
                    {
                        // One fully contains the other
                        // delete lower score
                        size_t delIdx = (matches[i].score >= matches[j].score) ? j : i;
                        deleted[delIdx] = true;

                        if (delIdx == i)
                            break; // rect i is gone, stop comparing i
                    }
                    else // INTERSECT_PARTIAL
                    {
                        if (interPts.size() < 3)
                            continue;

                        // MatchTool sorts intersection points before contourArea
                        cv::Point2f center(0.f, 0.f);
                        for (const auto& p : interPts) center += p;
                        center *= (1.f / interPts.size());

                        std::sort(interPts.begin(), interPts.end(),
                            [&center](const cv::Point2f& a, const cv::Point2f& b) {
                                return atan2(a.y - center.y, a.x - center.x) <
                                    atan2(b.y - center.y, b.x - center.x);
                            });

                        double interArea = std::fabs(cv::contourArea(interPts));
                        double overlapRatio = interArea / templArea;

                        if (overlapRatio > m_maxOverlap)
                        {
                            size_t delIdx = (matches[i].score >= matches[j].score) ? j : i;
                            deleted[delIdx] = true;

                            if (delIdx == i)
                                break;
                        }
                    }
                }
            }

            // Collect results
            std::vector<MatchInfo> result;
            result.reserve(maxTargets);

            for (size_t i = 0; i < matches.size(); ++i)
            {
                if (!deleted[i])
                {
                    result.push_back(matches[i]);
                    if ((int)result.size() >= maxTargets)
                        break;
                }
            }

            return result;
        }

        void FastNCCMatcherImpl::setAngleRange(double minA, double maxA) {
            m_minAngle = minA;
            m_maxAngle = maxA;
        }

        void FastNCCMatcherImpl::setScoreThreshold(double t) {
            m_scoreThreshold = std::max(0.0, std::min(1.0, t));
        }

        void FastNCCMatcherImpl::setMaxOverlap(double o) {
            m_maxOverlap = std::max(0.0, std::min(1.0, o));
        }

        double FastNCCMatcherImpl::getMinAngle() const { return m_minAngle; }
        double FastNCCMatcherImpl::getMaxAngle() const { return m_maxAngle; }
        double FastNCCMatcherImpl::getScoreThreshold() const { return m_scoreThreshold; }
        double FastNCCMatcherImpl::getMaxOverlap() const { return m_maxOverlap; }



        // Getter methods
        std::vector<cv::Mat> FastNCCMatcherImpl::getTemplatePyramid() const {
            return m_templatePyramid;
        }

        std::vector<cv::Scalar> FastNCCMatcherImpl::getVecTemplMean() const {
            return m_vecTemplMean;
        }

        std::vector<double> FastNCCMatcherImpl::getVecTemplNorm() const {
            return m_vecTemplNorm;
        }

        std::vector<double> FastNCCMatcherImpl::getVecTemplInvArea() const {
            return m_vecInvArea;
        }

        std::vector<bool> FastNCCMatcherImpl::getVecResultEqual1() const {
            return m_vecResultEqual1;
        }

        bool FastNCCMatcherImpl::getIsLearned() const {
            return m_isLearned;
        }

        int FastNCCMatcherImpl::getNumLevels() const {
            return num_levels;
        }

        int FastNCCMatcherImpl::getNumLevelsForSrc() const {
            return num_levels_for_src;
        }

        // Setter methods
        void FastNCCMatcherImpl::setNumLevels(int numLevels) {
            num_levels = numLevels;
        }

        void FastNCCMatcherImpl::setNumLevelsForSrc(int numLevels) {
            num_levels_for_src = numLevels;
        }

        void FastNCCMatcherImpl::setTemplatePyramid(const std::vector<cv::Mat>& templatePyramid) {
            m_templatePyramid = templatePyramid;
        }

        void FastNCCMatcherImpl::setVecTemplMean(const std::vector<cv::Scalar>& vecTemplMean) {
            m_vecTemplMean = vecTemplMean;
        }

        void FastNCCMatcherImpl::setVecTemplNorm(const std::vector<double>& templNorm) {
            m_vecTemplNorm = templNorm;
        }

        void FastNCCMatcherImpl::setVecTemplInvArea(const std::vector<double>& invArea) {
            m_vecInvArea = invArea;
        }

        void FastNCCMatcherImpl::setVecResultEqual1(const std::vector<bool>& resultEqual1) {
            m_vecResultEqual1 = resultEqual1;
        }
    } // namespace pattern_matching
} // namespace cv
