// ================================================================================================
// fast_ncc_matcher.h - Complete Header File
// ================================================================================================
#ifndef OPENCV_PATTERN_MATCHING_FAST_NCC_MATCHER_H
#define OPENCV_PATTERN_MATCHING_FAST_NCC_MATCHER_H
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>

namespace cv {
    namespace pattern_matching {
        struct MatchInfo {
            double score;
            double angle;
            double scale;    // isotropic shortcut; avg when aniso
            double scaleR;
            double scaleC;
            cv::Point2d position;
            int modelId;
            MatchInfo()
                : score(-1.0), angle(0.0), scale(1.0), scaleR(1.0), scaleC(1.0)
                , position(0, 0), modelId(0) {}
            MatchInfo(double s, double a, cv::Point2d p,
                      double sc = 1.0, int mid = 0, double scR = 1.0, double scC = 1.0)
                : score(s), angle(a), scale(sc), scaleR(scR), scaleC(scC)
                , position(p), modelId(mid) {}
        };

        class FastNCCMatcher : public cv::Algorithm {
        public:
            static cv::Ptr<FastNCCMatcher> create();

            static std::vector<MatchInfo> MatchTemplate(
                cv::InputArray source,
                cv::InputArray templateImg,
                double scoreThreshold = 0.7,
                double angleRange = 180.0,
                double maxOverlap = 0.5,
                int minReducedArea = 256,
                int maxTargets = 5
            );

            virtual bool setTemplate(cv::InputArray templateImg) = 0;
            virtual bool train(int numLevels, int minReducedArea = 256) = 0;
            // minPyramidLevel: stop refinement at this level (0 = refine to full resolution)
            virtual std::vector<MatchInfo> match(cv::InputArray source, int numLevels, int maxTargets = 5, int minPyramidLevel = 0) = 0;
            virtual void setAngleRange(double minAngle, double maxAngle) = 0;
            virtual void setScoreThreshold(double threshold) = 0;
            virtual void setMaxOverlap(double overlap) = 0;


            virtual double getMinAngle() const = 0;
            virtual double getMaxAngle() const = 0;
            virtual double getScoreThreshold() const = 0;
            virtual double getMaxOverlap() const = 0;


            virtual void setNumLevels(int numLevels) = 0;
            virtual int getNumLevels() const = 0;
            virtual void setNumLevelsForSrc(int numLevles) = 0;
            virtual int getNumLevelsForSrc() const = 0;

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

            virtual bool getIsLearned() const = 0;


            virtual ~FastNCCMatcher() {}
        };

        class FastNCCMatcherImpl : public FastNCCMatcher {
        public:
            FastNCCMatcherImpl();
            virtual ~FastNCCMatcherImpl();
            bool setTemplate(cv::InputArray templateImg) CV_OVERRIDE;
            bool train(int numLevels, int minReducedArea) CV_OVERRIDE;
            std::vector<MatchInfo> match(cv::InputArray source, int numLevels, int maxTargets, int minPyramidLevel) CV_OVERRIDE;
            void setAngleRange(double minAngle, double maxAngle) CV_OVERRIDE;
            void setScoreThreshold(double threshold) CV_OVERRIDE;
            void setMaxOverlap(double overlap) CV_OVERRIDE;


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
            cv::Mat m_templateImage;
            std::vector<cv::Mat> m_templatePyramid; // CV_32FC4
            std::vector<cv::Scalar> m_vecTemplMean; // mean per channel
            std::vector<double>     m_vecTemplNorm; //norm
            std::vector<double>     m_vecInvArea; // mean per channel
            std::vector<bool> m_vecResultEqual1;  // constant template flag
            int num_levels;
            int num_levels_for_src;
            bool m_isLearned;
            bool m_isTemplateSet;
            double m_minAngle;
            double m_maxAngle;
            double m_scoreThreshold;
            double m_maxOverlap;
            int m_borderColor;
            void buildPyramid(const cv::Mat& image, std::vector<cv::Mat>& pyramid, int numLevels, int minArea);
            std::vector<MatchInfo> filterMatches(std::vector<MatchInfo>& matches, int maxTargets);
            void buildSrcPyramid(const cv::Mat& image, std::vector<cv::Mat>& pyramid, int topLayer);
        };
    }
}
#endif
