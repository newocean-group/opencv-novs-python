// shape_based_matcher.h - Halcon-style shape-based matching.
// Custom edge-gradient correlation implementation (no longer depends on linemod).
#ifndef OPENCV_PATTERN_MATCHING_SHAPE_BASED_MATCHER_H
#define OPENCV_PATTERN_MATCHING_SHAPE_BASED_MATCHER_H

#include "opencv2/pattern_matching/fast_ncc_matcher.hpp"
#include <algorithm>
#include <cstdint>
#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

namespace cv {
    namespace pattern_matching {

        // Halcon Metric parameter equivalents (grayscale / single-channel).
        enum class ShapeMatchMetric {
            UsePolarity = 0,           // signed dot product (PatMax default)
            IgnoreGlobalPolarity = 1,  // max(score, score with model gradients flipped 180°)
            IgnoreLocalPolarity = 2,   // |dot| — match regardless of local contrast sign
        };

        // Halcon SubPixel parameter equivalents.
        enum class ShapeSubPixelMode {
            None = 0,           // integer pixel / angle step only
            Interpolation = 1,  // sub-pixel position via score interpolation (no angle refine)
            LeastSquares = 2,   // full 2D quadratic position + 1D angle parabola (default)
        };

        // Pass 0 (or <= 0) to setAngleStep for Halcon-style automatic step:
        //   step = atan(2 / max(template width, height)) converted to degrees.
        constexpr double kAutoAngleStep = 0.0;

        // Pass 0 to setNumPyramidLevels for Halcon-style automatic depth:
        //   floor(log2(min(width, height) / 16)), clamped to [1, 6].
        constexpr int kAutoNumPyramidLevels = 0;

        // ---------- Internal model representation (POD, suitable for save/load) ----------

        struct ShapeTemplateMeta {
            std::string name;
            int id;
            int refW;
            int refH;
            float originX;   // reference point in full-res template pixels
            float originY;
        };

        struct ShapeEdgeFeature {
            float dx, dy;     // offset from template top-left (post-pyramid, post-rotation).
                              // Kept as float so rotation + pyramid scaling stay subpixel-accurate.
            float gx, gy;     // unit gradient direction
        };

        struct ShapeModelLevel {
            int width;        // template bounding box at this pyramid level
            int height;
            float originX;    // model origin (subtract from match top-left to get reference pt)
            float originY;
            std::vector<ShapeEdgeFeature> features;
            std::vector<int> featOx;  // cached (int)lround(dx) for coarse find
            std::vector<int> featOy;

            void cacheFeatureOffsets() {
                const int n = (int)features.size();
                featOx.resize(n);
                featOy.resize(n);
                for (int i = 0; i < n; ++i) {
                    featOx[i] = (int)std::lround(features[i].dx);
                    featOy[i] = (int)std::lround(features[i].dy);
                }
                // Row-major feature order improves belt coarse cache locality (P13).
                std::vector<int> order(n);
                for (int i = 0; i < n; ++i) order[i] = i;
                std::stable_sort(order.begin(), order.end(), [this](int a, int b) {
                    if (featOy[a] != featOy[b]) return featOy[a] < featOy[b];
                    return featOx[a] < featOx[b];
                });
                if (order[0] == 0 && (n <= 1 || order[1] == 1)) return;
                std::vector<ShapeEdgeFeature> f2;
                std::vector<int> ox2, oy2;
                f2.reserve(n); ox2.reserve(n); oy2.reserve(n);
                for (int i : order) {
                    f2.push_back(features[i]);
                    ox2.push_back(featOx[i]);
                    oy2.push_back(featOy[i]);
                }
                features = std::move(f2);
                featOx = std::move(ox2);
                featOy = std::move(oy2);
            }
        };

        struct ShapePoseModel {
            float angle;
            float scale;      // isotropic average / legacy
            float scaleR;
            float scaleC;
            int templateId;
            std::vector<ShapeModelLevel> levels;
        };

        // Normalized gradient pyramid built from a search image (cached across finds).
        struct SourcePyramid {
            std::vector<cv::Mat> gray;
            std::vector<cv::Mat> nx;
            std::vector<cv::Mat> ny;
            std::vector<cv::Mat> mag;
        };

        // Halcon determine_shape_model_params output.
        struct ShapeModelParams {
            int numPyramidLevels;
            double angleStep;
            float minContrast;
            int numFeatures;
        };

        ShapeModelParams determineShapeModelParams(cv::InputArray templateImg,
                                                   cv::InputArray mask = cv::noArray());

        // ---------- Public matcher class ----------

        class CV_EXPORTS_W ShapeBasedMatcher : public cv::Algorithm {
        public:
            CV_WRAP static cv::Ptr<ShapeBasedMatcher> create();

            // Training inputs
            CV_WRAP virtual bool setTemplate(cv::InputArray templateImg, cv::InputArray mask) = 0;
            CV_WRAP virtual void clearTemplates() = 0;
            CV_WRAP virtual int addTemplate(cv::InputArray templateImg, cv::InputArray mask,
                                    const std::string& name = std::string()) = 0;
            CV_WRAP virtual int numModels() const = 0;
            CV_WRAP virtual int numPoseModels() const = 0;

            CV_WRAP virtual void setOrigin(double x, double y) = 0;
            CV_WRAP virtual double getOriginX() const = 0;
            CV_WRAP virtual double getOriginY() const = 0;
            CV_WRAP virtual void resetOrigin() = 0;

            // Extra offset added to find() position (external frame calibration).
            CV_WRAP virtual void setPositionBias(double x, double y) = 0;
            CV_WRAP virtual double getPositionBiasX() const = 0;
            CV_WRAP virtual double getPositionBiasY() const = 0;

            // Halcon §2.4.1 find hint: narrow belt search around expected (col, row).
            CV_WRAP virtual void setFindHint(double col, double row) = 0;
            CV_WRAP virtual void clearFindHint() = 0;
            CV_WRAP virtual bool hasFindHint() const = 0;
            CV_WRAP virtual void setTrackLastFind(bool enable) = 0;
            CV_WRAP virtual bool getTrackLastFind() const = 0;

            // Halcon find NumLevels second value: finest pyramid level to refine (0=L0, 1=L1, …).
            CV_WRAP virtual void setFindLowestPyramidLevel(int level) = 0;
            CV_WRAP virtual int getFindLowestPyramidLevel() const = 0;

            CV_WRAP virtual void setAngleRange(double minAngle, double maxAngle) = 0;
            CV_WRAP virtual void setAngleStep(double step) = 0;
            CV_WRAP virtual double getMinAngle() const = 0;
            CV_WRAP virtual double getMaxAngle() const = 0;
            CV_WRAP virtual double getAngleStep() const = 0;
            CV_WRAP virtual double getEffectiveAngleStep() const = 0;

            CV_WRAP virtual void setScaleRange(double minScale, double maxScale, double step) = 0;
            CV_WRAP virtual void setAnisoScaleRange(double minScaleR, double maxScaleR, double stepR,
                                            double minScaleC, double maxScaleC, double stepC) = 0;
            CV_WRAP virtual double getMinScaleR() const = 0;
            CV_WRAP virtual double getMaxScaleR() const = 0;
            CV_WRAP virtual double getScaleStepR() const = 0;
            CV_WRAP virtual double getMinScaleC() const = 0;
            CV_WRAP virtual double getMaxScaleC() const = 0;
            CV_WRAP virtual double getScaleStepC() const = 0;
            CV_WRAP virtual bool getAnisoScaleEnabled() const = 0;

            CV_WRAP virtual bool setContour(cv::InputArray points, cv::InputArray gradients) = 0;
            CV_WRAP virtual void setContourRefSize(int width, int height) = 0;
            CV_WRAP virtual void setCropTemplateToMask(bool enable) = 0;
            CV_WRAP virtual bool getCropTemplateToMask() const = 0;

            CV_WRAP virtual void setUseIcpRefine(bool enable) = 0;
            CV_WRAP virtual bool getUseIcpRefine() const = 0;
            CV_WRAP virtual void setIcpMaxIterations(int n) = 0;
            CV_WRAP virtual int getIcpMaxIterations() const = 0;

            CV_WRAP virtual double getMinScale() const = 0;
            CV_WRAP virtual double getMaxScale() const = 0;
            CV_WRAP virtual double getScaleStep() const = 0;

            // Match-time tuning (Halcon-like)
            CV_WRAP virtual void setNumPyramidLevels(int n) = 0;
            CV_WRAP virtual int getNumPyramidLevels() const = 0;
            CV_WRAP virtual int getEffectiveNumPyramidLevels() const = 0;
            CV_WRAP virtual void setGreediness(double g) = 0;       // 0..1 (0=thorough, 1=aggressive)
            CV_WRAP virtual double getGreediness() const = 0;
            CV_WRAP virtual void setMaxOverlap(double overlap) = 0; // 0..1 rotated-rect IoU NMS
            CV_WRAP virtual double getMaxOverlap() const = 0;
            CV_WRAP virtual void setUseSubPixel(bool enable) = 0;
            CV_WRAP virtual bool getUseSubPixel() const = 0;
            CV_WRAP virtual void setSubPixelMode(ShapeSubPixelMode mode) = 0;
            CV_WRAP virtual ShapeSubPixelMode getSubPixelMode() const = 0;
            CV_WRAP virtual void setFindTimeoutMs(int timeoutMs) = 0;
            CV_WRAP virtual int getFindTimeoutMs() const = 0;
            CV_WRAP virtual void setMinContrast(float c) = 0;       // gradient magnitude floor for edges
            CV_WRAP virtual float getMinContrast() const = 0;
            CV_WRAP virtual void setNumFeatures(int n) = 0;         // edges kept per pyramid level
            CV_WRAP virtual int getNumFeatures() const = 0;

            CV_WRAP virtual void setMetric(ShapeMatchMetric metric) = 0;
            CV_WRAP virtual ShapeMatchMetric getMetric() const = 0;
            // Back-compat wrappers around setMetric(UsePolarity / IgnoreLocalPolarity).
            CV_WRAP virtual void setUsePolarity(bool enable) = 0;
            CV_WRAP virtual bool getUsePolarity() const = 0;

            // Model lifecycle
            CV_WRAP virtual bool train() = 0;
            CV_WRAP virtual bool saveModel(const std::string& path) const = 0;
            CV_WRAP virtual bool loadModel(const std::string& path) = 0;

            // Matching
            CV_WRAP virtual std::vector<MatchInfo> find(cv::InputArray source,
                                                float scoreThreshold,
                                                int maxTargets,
                                                cv::Rect searchRegion = cv::Rect(),
                                                cv::InputArray searchMask = cv::noArray()) = 0;

            CV_WRAP virtual bool isTrained() const = 0;
            CV_WRAP virtual int numTemplates() const = 0; // alias for numModels()

            CV_WRAP virtual int getFeatures(double angle, int level, cv::OutputArray out,
                                    double scaleR = 1.0, double scaleC = 1.0,
                                    int modelIndex = 0) const = 0;

            // Halcon get_shape_model_contours: N×2 CV_32F contour points.
            // When transformToImage is true, applies pose (poseX, poseY, poseAngleDeg).
            CV_WRAP virtual int getShapeModelContours(double angle, int level, cv::OutputArray out,
                                              double scaleR, double scaleC, int modelIndex,
                                              double poseX, double poseY, double poseAngleDeg,
                                              bool transformToImage) const = 0;

            // C++ only (vector outputs not wrapped for Python).
            virtual int inspectModel(int modelIndex,
                                     std::vector<cv::Mat>& levelImages,
                                     std::vector<int>& featureCounts) const = 0;

            virtual ~ShapeBasedMatcher() {}
        };

        class ShapeBasedMatcherImpl : public ShapeBasedMatcher {
        public:
            ShapeBasedMatcherImpl();
            virtual ~ShapeBasedMatcherImpl();

            bool setTemplate(cv::InputArray templateImg, cv::InputArray mask) CV_OVERRIDE;
            void clearTemplates() CV_OVERRIDE;
            int addTemplate(cv::InputArray templateImg, cv::InputArray mask,
                            const std::string& name) CV_OVERRIDE;
            int numModels() const CV_OVERRIDE;
            int numPoseModels() const CV_OVERRIDE;

            void setOrigin(double x, double y) CV_OVERRIDE;
            double getOriginX() const CV_OVERRIDE;
            double getOriginY() const CV_OVERRIDE;
            void resetOrigin() CV_OVERRIDE;

            void setPositionBias(double x, double y) CV_OVERRIDE;
            double getPositionBiasX() const CV_OVERRIDE;
            double getPositionBiasY() const CV_OVERRIDE;

            void setFindHint(double col, double row) CV_OVERRIDE;
            void clearFindHint() CV_OVERRIDE;
            bool hasFindHint() const CV_OVERRIDE;
            void setTrackLastFind(bool enable) CV_OVERRIDE;
            bool getTrackLastFind() const CV_OVERRIDE;

            void setFindLowestPyramidLevel(int level) CV_OVERRIDE;
            int getFindLowestPyramidLevel() const CV_OVERRIDE;

            void setAngleRange(double minAngle, double maxAngle) CV_OVERRIDE;
            void setAngleStep(double step) CV_OVERRIDE;
            double getMinAngle() const CV_OVERRIDE;
            double getMaxAngle() const CV_OVERRIDE;
            double getAngleStep() const CV_OVERRIDE;
            double getEffectiveAngleStep() const CV_OVERRIDE;

            void setScaleRange(double minScale, double maxScale, double step) CV_OVERRIDE;
            void setAnisoScaleRange(double minScaleR, double maxScaleR, double stepR,
                                    double minScaleC, double maxScaleC, double stepC) CV_OVERRIDE;
            double getMinScaleR() const CV_OVERRIDE;
            double getMaxScaleR() const CV_OVERRIDE;
            double getScaleStepR() const CV_OVERRIDE;
            double getMinScaleC() const CV_OVERRIDE;
            double getMaxScaleC() const CV_OVERRIDE;
            double getScaleStepC() const CV_OVERRIDE;
            bool getAnisoScaleEnabled() const CV_OVERRIDE;

            bool setContour(cv::InputArray points, cv::InputArray gradients) CV_OVERRIDE;
            void setContourRefSize(int width, int height) CV_OVERRIDE;
            void setCropTemplateToMask(bool enable) CV_OVERRIDE;
            bool getCropTemplateToMask() const CV_OVERRIDE;

            void setUseIcpRefine(bool enable) CV_OVERRIDE;
            bool getUseIcpRefine() const CV_OVERRIDE;
            void setIcpMaxIterations(int n) CV_OVERRIDE;
            int getIcpMaxIterations() const CV_OVERRIDE;

            double getMinScale() const CV_OVERRIDE;
            double getMaxScale() const CV_OVERRIDE;
            double getScaleStep() const CV_OVERRIDE;

            void setNumPyramidLevels(int n) CV_OVERRIDE;
            int getNumPyramidLevels() const CV_OVERRIDE;
            int getEffectiveNumPyramidLevels() const CV_OVERRIDE;
            void setGreediness(double g) CV_OVERRIDE;
            double getGreediness() const CV_OVERRIDE;
            void setMaxOverlap(double overlap) CV_OVERRIDE;
            double getMaxOverlap() const CV_OVERRIDE;
            void setUseSubPixel(bool enable) CV_OVERRIDE;
            bool getUseSubPixel() const CV_OVERRIDE;
            void setSubPixelMode(ShapeSubPixelMode mode) CV_OVERRIDE;
            ShapeSubPixelMode getSubPixelMode() const CV_OVERRIDE;
            void setFindTimeoutMs(int timeoutMs) CV_OVERRIDE;
            int getFindTimeoutMs() const CV_OVERRIDE;
            void setMinContrast(float c) CV_OVERRIDE;
            float getMinContrast() const CV_OVERRIDE;
            void setNumFeatures(int n) CV_OVERRIDE;
            int getNumFeatures() const CV_OVERRIDE;

            void setMetric(ShapeMatchMetric metric) CV_OVERRIDE;
            ShapeMatchMetric getMetric() const CV_OVERRIDE;
            void setUsePolarity(bool enable) CV_OVERRIDE;
            bool getUsePolarity() const CV_OVERRIDE;

            bool train() CV_OVERRIDE;
            bool saveModel(const std::string& path) const CV_OVERRIDE;
            bool loadModel(const std::string& path) CV_OVERRIDE;

            std::vector<MatchInfo> find(cv::InputArray source,
                                        float scoreThreshold,
                                        int maxTargets,
                                        cv::Rect searchRegion = cv::Rect(),
                                        cv::InputArray searchMask = cv::noArray()) CV_OVERRIDE;

            bool isTrained() const CV_OVERRIDE;
            int numTemplates() const CV_OVERRIDE;
            int getFeatures(double angle, int level, cv::OutputArray out,
                            double scaleR = 1.0, double scaleC = 1.0,
                            int modelIndex = 0) const CV_OVERRIDE;
            int getShapeModelContours(double angle, int level, cv::OutputArray out,
                                      double scaleR, double scaleC, int modelIndex,
                                      double poseX, double poseY, double poseAngleDeg,
                                      bool transformToImage) const CV_OVERRIDE;
            int inspectModel(int modelIndex,
                             std::vector<cv::Mat>& levelImages,
                             std::vector<int>& featureCounts) const CV_OVERRIDE;

        private:
            struct PendingTemplate {
                std::string name;
                cv::Mat image;
                cv::Mat mask;
                double originX;
                double originY;
                bool originSet;
                bool fromContour;
                int contourRefW;
                int contourRefH;
                std::vector<ShapeEdgeFeature> contourFeatures; // template-local pixels
            };

            struct BaseLevel {
                cv::Size sz;
                std::vector<ShapeEdgeFeature> features;
                cv::Point2f origin;
                int width, height;
            };

            int findBestPoseModel(double angle, double scaleR, double scaleC,
                                  int modelIndex) const;

            bool prepareMask(cv::InputArray templateImg, cv::InputArray mask,
                             cv::Mat& outMask) const;
            bool parseContourInput(cv::InputArray points, cv::InputArray gradients,
                                   PendingTemplate& pt) const;
            void buildScalePairs(std::vector<std::pair<float,float>>& out) const;
            bool trainOneImage(const PendingTemplate& pt, int templateId, cv::Mat gray);
            bool trainOneContour(const PendingTemplate& pt, int templateId);
            bool finalizePoseModel(ShapePoseModel& pm, BaseLevel* baseLevels, int nLevels,
                                   float ang, float scR, float scC, int templateId);
            bool trainOne(const PendingTemplate& pt, int templateId);
            bool validatePyramidDepthSelfMatch(const cv::Mat& trainGray);

            void clearSceneGradientCache();
            void ensureScenePyramid(const cv::Mat& srcGray, int nLevels, int throughLevel,
                                    SourcePyramid& out, bool l0OnlyMode = false);

            void icpRefinePose(const cv::Mat& nx, const cv::Mat& ny, const cv::Mat& mag,
                               float& x, float& y, float& angleDeg,
                               const ShapeModelLevel& ml,
                               float magFloor, ShapeMatchMetric metric) const;

            // ---- inputs ----
            std::vector<PendingTemplate> m_pending;
            cv::Mat m_templateImg;  // kept for inspect after single SetTemplate train
            cv::Mat m_mask;
            double m_originX;
            double m_originY;
            bool   m_originCustom;
            double m_positionBiasX;
            double m_positionBiasY;
            double m_findHintCol;
            double m_findHintRow;
            double m_findHintScore;
            bool   m_hasFindHint;
            bool   m_trackLastFind;
            int    m_findLowestPyramidLevel; // 0=auto (L0; L1 when belt track+hint)

            // ---- tuning ----
            double m_minAngle;
            double m_maxAngle;
            double m_angleStep;          // user request; <= 0 means auto at train time
            double m_effectiveAngleStep; // actual step used after train()
            double m_minScale;
            double m_maxScale;
            double m_scaleStep;
            double m_minScaleR;
            double m_maxScaleR;
            double m_scaleStepR;
            double m_minScaleC;
            double m_maxScaleC;
            double m_scaleStepC;
            bool   m_anisoScaleEnabled;
            int    m_numPyramidLevels;
            double m_greediness;
            double m_maxOverlap;
            ShapeSubPixelMode m_subPixelMode;
            int    m_findTimeoutMs;
            bool   m_useIcpRefine;
            int    m_icpMaxIterations;
            float  m_minContrast;
            int    m_numFeatures;
            ShapeMatchMetric m_metric;
            bool   m_metricUserSet;
            int    m_contourRefW;
            int    m_contourRefH;
            bool   m_cropTemplateToMask;
            bool   m_forceStrictPyramidLevels = false;

            // Scene gradient cache (same image searched repeatedly in production).
            int      m_scW = 0;
            int      m_scH = 0;
            std::uint64_t m_scFp = 0;
            int      m_scLevels = 0;
            int      m_scBuiltThrough = -1;
            SourcePyramid m_scenePyramid;

            // ---- trained model ----
            std::vector<ShapeTemplateMeta> m_templateMeta;
            std::vector<ShapePoseModel> m_models;
            int  m_actualNumLevels;
            int  m_templateRefW;
            int  m_templateRefH;
            bool m_isTrained;
        };

    } // namespace pattern_matching
} // namespace cv

#endif
