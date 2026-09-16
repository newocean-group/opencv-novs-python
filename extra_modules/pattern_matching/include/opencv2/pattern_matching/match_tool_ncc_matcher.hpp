// match_tool_ncc_matcher.h — MatchTool-style SIMD NCC (grayscale 8U pyramid), same API shape as FastNCCMatcher
#ifndef OPENCV_PATTERN_MATCHING_MATCH_TOOL_NCC_MATCHER_H
#define OPENCV_PATTERN_MATCHING_MATCH_TOOL_NCC_MATCHER_H

#include "fast_ncc_matcher.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>

namespace cv {
    namespace pattern_matching {

        class CV_EXPORTS_W MatchToolNCCMatcher : public cv::Algorithm {
        public:
            CV_WRAP static cv::Ptr<MatchToolNCCMatcher> create();

            CV_WRAP static std::vector<MatchInfo> MatchTemplate(
                cv::InputArray source,
                cv::InputArray templateImg,
                int numLevels,
                double scoreThreshold,
                double angleRange,
                double maxOverlap,
                int minReducedArea,
                int maxTargets,
                bool useSubPixel = false
            );

            CV_WRAP virtual bool setTemplate(cv::InputArray templateImg) = 0;
            CV_WRAP virtual bool train(int numLevels, int minReducedArea = 256) = 0;
            // minPyramidLevel: stop refinement at this level (0 = refine to full resolution)
            CV_WRAP virtual std::vector<MatchInfo> match(cv::InputArray source, int numLevels, int maxTargets = 5, int minPyramidLevel = 0) = 0;
            CV_WRAP virtual void setAngleRange(double minAngle, double maxAngle) = 0;
            CV_WRAP virtual void setScoreThreshold(double threshold) = 0;
            CV_WRAP virtual void setMaxOverlap(double overlap) = 0;

            CV_WRAP virtual void setUseSubPixel(bool enable) = 0;
            CV_WRAP virtual bool getUseSubPixel() const = 0;

            CV_WRAP virtual double getMinAngle() const = 0;
            CV_WRAP virtual double getMaxAngle() const = 0;
            CV_WRAP virtual double getScoreThreshold() const = 0;
            CV_WRAP virtual double getMaxOverlap() const = 0;

            CV_WRAP virtual void setNumLevels(int numLevels) = 0;
            CV_WRAP virtual int getNumLevels() const = 0;
            CV_WRAP virtual void setNumLevelsForSrc(int numLevels) = 0;
            CV_WRAP virtual int getNumLevelsForSrc() const = 0;

            virtual void setTemplatePyramid(const std::vector<cv::Mat>& templatePyramid) = 0;
            virtual void setVecTemplMean(const std::vector<cv::Scalar>& vecTemplMean) = 0;
            virtual void setVecTemplNorm(const std::vector<double>& templNorm) = 0;
            virtual void setVecTemplInvArea(const std::vector<double>& invArea) = 0;
            virtual void setVecResultEqual1(const std::vector<bool>& resultEqual1) = 0;

            virtual std::vector<cv::Mat> getTemplatePyramid() const = 0;
            virtual std::vector<cv::Scalar> getVecTemplMean() const = 0;
            virtual std::vector<double> getVecTemplNorm() const = 0;
            virtual std::vector<double> getVecTemplInvArea() const = 0;
            virtual std::vector<bool> getVecResultEqual1() const = 0;

            CV_WRAP virtual bool getIsLearned() const = 0;

            virtual ~MatchToolNCCMatcher() {}
        };

        class MatchToolNCCMatcherImpl : public MatchToolNCCMatcher {
        public:
            MatchToolNCCMatcherImpl();
            virtual ~MatchToolNCCMatcherImpl();

            bool setTemplate(cv::InputArray templateImg) CV_OVERRIDE;
            bool train(int numLevels, int minReducedArea) CV_OVERRIDE;
            std::vector<MatchInfo> match(cv::InputArray source, int numLevels, int maxTargets, int minPyramidLevel) CV_OVERRIDE;
            void setAngleRange(double minAngle, double maxAngle) CV_OVERRIDE;
            void setScoreThreshold(double threshold) CV_OVERRIDE;
            void setMaxOverlap(double overlap) CV_OVERRIDE;

            void setUseSubPixel(bool enable) CV_OVERRIDE;
            bool getUseSubPixel() const CV_OVERRIDE;

            double getMinAngle() const CV_OVERRIDE;
            double getMaxAngle() const CV_OVERRIDE;
            double getScoreThreshold() const CV_OVERRIDE;
            double getMaxOverlap() const CV_OVERRIDE;

            void setNumLevels(int numLevels) CV_OVERRIDE;
            int getNumLevels() const CV_OVERRIDE;
            void setNumLevelsForSrc(int numLevels) CV_OVERRIDE;
            int getNumLevelsForSrc() const CV_OVERRIDE;

            void setTemplatePyramid(const std::vector<cv::Mat>& templatePyramid) CV_OVERRIDE;
            void setVecTemplMean(const std::vector<cv::Scalar>& vecTemplMean) CV_OVERRIDE;
            void setVecTemplNorm(const std::vector<double>& templNorm) CV_OVERRIDE;
            void setVecTemplInvArea(const std::vector<double>& invArea) CV_OVERRIDE;
            void setVecResultEqual1(const std::vector<bool>& resultEqual1) CV_OVERRIDE;

            std::vector<cv::Mat> getTemplatePyramid() const CV_OVERRIDE;
            std::vector<cv::Scalar> getVecTemplMean() const CV_OVERRIDE;
            std::vector<double> getVecTemplNorm() const CV_OVERRIDE;
            std::vector<double> getVecTemplInvArea() const CV_OVERRIDE;
            std::vector<bool> getVecResultEqual1() const CV_OVERRIDE;

            bool getIsLearned() const CV_OVERRIDE;

        private:
            // Pre-rotated template + its precomputed NCC normalization terms.
            struct RotatedCacheEntry {
                cv::Mat templ;
                cv::Scalar mean;
                double norm;
                double invArea;
                bool resultEqual1;
            };

            cv::Mat m_templateGray8u;
            std::vector<cv::Mat> m_templatePyramid;
            std::vector<cv::Scalar> m_vecTemplMean;
            std::vector<double> m_vecTemplNorm;
            std::vector<double> m_vecInvArea;
            std::vector<bool> m_vecResultEqual1;
            int num_levels;
            int num_levels_for_src;
            bool m_isLearned;
            bool m_isTemplateSet;
            double m_minAngle;
            double m_maxAngle;
            double m_scoreThreshold;
            double m_maxOverlap;
            int m_borderColor;
            bool m_useSubPixel;

            // Pre-computed rotation cache. Cleared in setTemplate(), built in train().
            // Angle range is fixed at train time; setAngleRange before train/match.
            // Indexed [pyramidLevel][angleIdx].
            std::vector<std::vector<RotatedCacheEntry>> m_rotCache;
            std::vector<std::vector<double>> m_rotCacheAngles;
            std::vector<double> m_rotCacheStep;
            double m_rotCacheMinAngle;
            double m_rotCacheMaxAngle;
            bool m_rotCacheValid;

            void buildPyramid(const cv::Mat& image, std::vector<cv::Mat>& pyramid, int numLevels, int minArea);
            void buildSrcPyramid(const cv::Mat& image, std::vector<cv::Mat>& pyramid, int topLayer);
            std::vector<MatchInfo> filterMatches(std::vector<MatchInfo>& matches, int maxTargets);

            void buildRotationCache();
            void invalidateRotationCache();
            int findNearestCachedAngle(int level, double angle) const;
        };

    } // namespace pattern_matching
} // namespace cv

#endif
