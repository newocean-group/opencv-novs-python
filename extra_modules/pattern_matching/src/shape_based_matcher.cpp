// shape_based_matcher.cpp - Halcon/PatMax-style shape-based matching.
//
// Algorithm overview:
//   1. Training (per pyramid level, once):
//      - pyrDown the template + mask
//      - Erode the mask 1px so the artificial template-border gradient (the
//        transition from object pixels to the black padding) is NOT picked up
//        as an edge feature. This is essential: otherwise the matcher keys on
//        the template's rectangular frame instead of the object.
//      - Sobel gradients, non-max suppression along gradient direction
//      - Pick the N strongest, well-scattered edge features
//      - Store as { offset_from_top_left (float), unit_gradient_direction }
//   2. Training (per angle):
//      - Rotate the level-0 features (positions and gradient directions both),
//        keeping positions in floating point so fine angle steps stay accurate.
//   3. Matching:
//      - Build source pyramid; per level compute Sobel, |grad|, and the unit
//        gradient field (gx/|g|, gy/|g|).
//      - At the topmost (smallest) level, exhaustively scan the source for every
//        angle model. Score = (1/N)*sum dot(t_grad, s_grad) with Halcon polarity rules.
//        MinScore is enforced at every pyramid level during refinement (Halcon §3.1.3).
//      - Spatial NMS on top-level candidates.
//      - Walk back down the pyramid refining each candidate in a small ROI,
//        sampling the gradient field with bilinear interpolation.
//      - Subpixel: 2D-quadratic on position + 1D-parabola on angle.
//      - Rotated-rect IoU NMS for final output.

#include "shape_based_matcher.h"
#include "opencvsharp_license_guard.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <list>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define SHAPE_HAS_SSE2 1
#include <emmintrin.h>
#endif
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <xmmintrin.h>
#endif

#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
#define SHAPE_HAS_AVX2 1
#include <immintrin.h>
#endif

namespace cv {
namespace pattern_matching {

namespace {

constexpr double kDeg2Rad = CV_PI / 180.0;

// --- small utils --------------------------------------------------------------

cv::Mat ensureGray8u(const cv::Mat& src) {
    if (src.empty()) return src;
    if (src.type() == CV_8UC1) return src;
    cv::Mat out;
    if (src.channels() == 3) cv::cvtColor(src, out, cv::COLOR_BGR2GRAY);
    else if (src.channels() == 4) cv::cvtColor(src, out, cv::COLOR_BGRA2GRAY);
    else src.convertTo(out, CV_8U);
    return out;
}

double parabolicPeakOffset(double sm, double s0, double sp) {
    const double denom = (sm - 2.0 * s0 + sp);
    if (std::abs(denom) < 1e-12) return 0.0;
    double off = 0.5 * (sm - sp) / denom;
    if (off < -1.0) off = -1.0;
    if (off > 1.0) off = 1.0;
    return off;
}

// Halcon least_squares: 2D quadratic peak on 3×3 score patch (center at v[1][1]).
void quadraticSubpixelOffset2D(const float v[3][3], double& ox, double& oy) {
    ox = parabolicPeakOffset(v[1][0], v[1][1], v[1][2]);
    oy = parabolicPeakOffset(v[0][1], v[1][1], v[2][1]);
    const float c = v[1][1];
    const float dxx = v[1][0] - 2.f * c + v[1][2];
    const float dyy = v[0][1] - 2.f * c + v[2][1];
    const float dxy = 0.25f * (v[0][0] - v[0][2] - v[2][0] + v[2][2]);
    const float det = dxx * dyy - dxy * dxy;
    if (std::fabs(det) > 1e-8f) {
        const float dx1 = 0.5f * (v[1][2] - v[1][0]);
        const float dy1 = 0.5f * (v[2][1] - v[0][1]);
        ox = -(double)(dyy * dx1 - dxy * dy1) / (double)det;
        oy = -(double)(-dxy * dx1 + dxx * dy1) / (double)det;
        if (ox < -1.0) ox = -1.0;
        if (ox > 1.0) ox = 1.0;
        if (oy < -1.0) oy = -1.0;
        if (oy > 1.0) oy = 1.0;
    }
}

double rotatedRectIoU(const cv::RotatedRect& a, const cv::RotatedRect& b) {
    std::vector<cv::Point2f> inter;
    int type = cv::rotatedRectangleIntersection(a, b, inter);
    if (type == cv::INTERSECT_NONE || inter.size() < 3) return 0.0;
    double interArea = std::fabs(cv::contourArea(inter));
    double areaA = (double)a.size.width * a.size.height;
    double areaB = (double)b.size.width * b.size.height;
    double unionArea = areaA + areaB - interArea;
    if (unionArea <= 0.0) return 0.0;
    return interArea / unionArea;
}

struct FindTimer {
    int timeoutMs;
    std::chrono::steady_clock::time_point deadline;
    std::atomic<bool> expired{false};

    explicit FindTimer(int ms) : timeoutMs(ms) {
        if (timeoutMs > 0)
            deadline = std::chrono::steady_clock::now() +
                       std::chrono::milliseconds(timeoutMs);
    }

    bool check() {
        if (timeoutMs <= 0) return false;
        if (expired.load(std::memory_order_relaxed)) return true;
        if (std::chrono::steady_clock::now() >= deadline) {
            expired.store(true, std::memory_order_relaxed);
            return true;
        }
        return false;
    }
};

// --- edge feature extraction --------------------------------------------------
//
// Sobel gradient + non-max suppression along the gradient direction. We then
// pick the top-N features by magnitude, with a simple uniform-scatter pass so
// they aren't all clumped along one strong edge.

void extractEdgeFeatures(
    const cv::Mat& gray8u,
    const cv::Mat& mask,         // 8U, may be empty
    int    numFeatures,
    float  minContrast,
    int    templateWidth,
    int    templateHeight,
    std::vector<ShapeEdgeFeature>& out)
{
    out.clear();
    if (gray8u.empty()) return;

    // Erode the mask by 1px on larger templates. The Sobel kernel is 3x3, so the
    // gradient at any pixel adjacent to the masked-off / padded region is contaminated
    // by that transition. For small ROIs (circle/ellipse) skip erosion — it removes
    // the only usable edge ring.
    cv::Mat useMask;
    if (!mask.empty()) {
        int erodePx = 1;
        cv::Rect roi = cv::boundingRect(mask);
        if (std::min(roi.width, roi.height) <= 40)
            erodePx = 0;
        if (erodePx > 0)
            cv::erode(mask, useMask, cv::Mat(), cv::Point(-1, -1), erodePx);
        else
            useMask = mask;
    }

    cv::Mat gx, gy;
    cv::Sobel(gray8u, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray8u, gy, CV_32F, 0, 1, 3);
    cv::Mat mag;
    cv::magnitude(gx, gy, mag);

    const int H = mag.rows;
    const int W = mag.cols;

    // NMS along discrete 4-direction.
    // Collect candidates (mag, x, y, gx, gy).
    struct Cand {
        float m;
        int x, y;
        float gx, gy;
    };
    std::vector<Cand> cands;
    cands.reserve(8192);

    auto collectCands = [&](float contrastFloor) {
        cands.clear();
        for (int y = 1; y < H - 1; ++y) {
            const float* mrow  = mag.ptr<float>(y);
            const float* mrowU = mag.ptr<float>(y - 1);
            const float* mrowD = mag.ptr<float>(y + 1);
            const float* gxrow = gx.ptr<float>(y);
            const float* gyrow = gy.ptr<float>(y);
            const uchar* mkrow = useMask.empty() ? nullptr : useMask.ptr<uchar>(y);

            for (int x = 1; x < W - 1; ++x) {
                if (mkrow && mkrow[x] == 0) continue;
                float m = mrow[x];
                if (m < contrastFloor) continue;

                float gxv = gxrow[x];
                float gyv = gyrow[x];

                float absx = std::abs(gxv), absy = std::abs(gyv);
                float m1, m2;
                if (absx > 2.414f * absy) {
                    m1 = mrow[x - 1]; m2 = mrow[x + 1];
                } else if (absy > 2.414f * absx) {
                    m1 = mrowU[x]; m2 = mrowD[x];
                } else if ((gxv > 0) == (gyv > 0)) {
                    m1 = mrowU[x - 1]; m2 = mrowD[x + 1];
                } else {
                    m1 = mrowU[x + 1]; m2 = mrowD[x - 1];
                }

                if (m >= m1 && m >= m2) {
                    Cand c;
                    c.m = m;
                    c.x = x;
                    c.y = y;
                    c.gx = gxv;
                    c.gy = gyv;
                    cands.push_back(c);
                }
            }
        }
    };

    collectCands(minContrast);
    if (cands.empty() && minContrast > 4.f)
        collectCands(minContrast * 0.5f);

    // Small circular/ring templates: Sobel+NMS may miss the edge ring; try Canny.
    if (cands.empty() && !useMask.empty()) {
        cv::Rect roi = cv::boundingRect(useMask);
        if (std::min(roi.width, roi.height) <= 64) {
            cv::Mat edges;
            const double cn = std::max(5.0, (double)minContrast);
            cv::Canny(gray8u, edges, cn, cn * 2.0, 3, true);
            if (!useMask.empty())
                cv::bitwise_and(edges, useMask, edges);
            for (int y = 1; y < H - 1; ++y) {
                const uchar* erow = edges.ptr<uchar>(y);
                const float* gxrow = gx.ptr<float>(y);
                const float* gyrow = gy.ptr<float>(y);
                for (int x = 1; x < W - 1; ++x) {
                    if (erow[x] == 0) continue;
                    float m = mag.at<float>(y, x);
                    if (m < minContrast * 0.25f) continue;
                    Cand c;
                    c.m = m;
                    c.x = x;
                    c.y = y;
                    c.gx = gxrow[x];
                    c.gy = gyrow[x];
                    cands.push_back(c);
                }
            }
        }
    }

    if (cands.empty()) return;

    // Sort by magnitude desc.
    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b) { return a.m > b.m; });

    // Scatter selection: take from the strongest down, but only keep a candidate
    // if it isn't within `minDist` of an already-kept one. Aim for `numFeatures`
    // total; if we don't fill up, relax minDist progressively.
    const int targetN = std::min<int>(numFeatures, (int)cands.size());
    if (targetN <= 0) return;

    // Start with a distance derived from template area / target features.
    const double templArea = (double)templateWidth * templateHeight;
    double minDist = std::sqrt(templArea / std::max(8, targetN)) * 0.5;
    if (minDist < 1.0) minDist = 1.0;
    double minDist2 = minDist * minDist;

    std::vector<ShapeEdgeFeature> picked;
    picked.reserve(targetN);

    // Keep trying with shrinking minDist until we have enough features (or run out).
    for (int attempt = 0; attempt < 4 && (int)picked.size() < targetN; ++attempt) {
        picked.clear();
        for (const auto& c : cands) {
            bool ok = true;
            for (const auto& p : picked) {
                double dx = c.x - p.dx;
                double dy = c.y - p.dy;
                if (dx * dx + dy * dy < minDist2) { ok = false; break; }
            }
            if (ok) {
                ShapeEdgeFeature f;
                f.dx = (float)c.x;
                f.dy = (float)c.y;
                float inv = 1.0f / c.m;
                f.gx = c.gx * inv;
                f.gy = c.gy * inv;
                picked.push_back(f);
                if ((int)picked.size() >= targetN) break;
            }
        }
        // Halve the spacing for next attempt.
        minDist  *= 0.5;
        minDist2  = minDist * minDist;
    }

    out = std::move(picked);
}

// Rotate features (positions and gradient directions) around a centre. Positions
// stay in floating point so that fine angle steps don't quantize to the pixel grid.
void rotateFeatures(
    const std::vector<ShapeEdgeFeature>& src,
    cv::Point2f center,
    float angleDeg,
    std::vector<ShapeEdgeFeature>& out)
{
    const float rad = static_cast<float>(angleDeg * kDeg2Rad);
    const float c = std::cos(rad);
    const float s = std::sin(rad);
    out.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        const ShapeEdgeFeature& f = src[i];
        ShapeEdgeFeature& r = out[i];
        const float dx = f.dx - center.x;
        const float dy = f.dy - center.y;
        r.dx = c * dx - s * dy + center.x;
        r.dy = s * dx + c * dy + center.y;
        r.gx = c * f.gx - s * f.gy;
        r.gy = s * f.gx + c * f.gy;
    }
}

// --- source pyramid + normalized gradient cache -------------------------------

static void computeGradLevel(const cv::Mat& gray, cv::Mat& nx, cv::Mat& ny, cv::Mat& mag);

static std::uint64_t sceneFingerprint(const cv::Mat& gray)
{
    std::uint64_t h = (std::uint64_t)(unsigned)gray.cols
                    ^ ((std::uint64_t)(unsigned)gray.rows << 20);
    if (gray.empty()) return h;
    const int rowStep = std::max(1, gray.rows / 32);
    const int colStep = std::max(1, gray.cols / 32);
    for (int y = 0; y < gray.rows; y += rowStep) {
        const uchar* row = gray.ptr(y);
        for (int x = 0; x < gray.cols; x += colStep)
            h = h * 1315423911u + row[x];
    }
    return h;
}

// Shared across matcher instances: keyed by scene fingerprint (bomach: many templates × few scenes).
struct CachedSceneEntry {
    int w = 0, h = 0, levels = 0, builtThrough = -1;
    std::uint64_t fp = 0;
    SourcePyramid pyramid;
};

struct GlobalScenePyramidCache {
    std::mutex mu;
    static constexpr size_t kMaxEntries = 48;
    std::unordered_map<std::uint64_t, CachedSceneEntry> entries;
    std::list<std::uint64_t> lru;
};

static GlobalScenePyramidCache g_sceneCache;

static void resizeSourcePyramid(SourcePyramid& sp, int nLevels)
{
    if (sp.nx.size() == (size_t)nLevels) return;
    sp.gray.assign(nLevels, cv::Mat());
    sp.nx.assign(nLevels, cv::Mat());
    sp.ny.assign(nLevels, cv::Mat());
    sp.mag.assign(nLevels, cv::Mat());
}

// Gray pyramid only (no Sobel) — L0 grad is skipped when fine search uses local ROI Sobel.
static void ensureCachedGrayLevel(SourcePyramid& sp, int nLevels, int level)
{
    if (level < 0 || level >= nLevels) return;
    resizeSourcePyramid(sp, nLevels);
    if (!sp.gray[level].empty()) return;
    if (level > 0) {
        ensureCachedGrayLevel(sp, nLevels, level - 1);
        cv::pyrDown(sp.gray[level - 1], sp.gray[level]);
    }
}

static void ensureCachedGradLevel(SourcePyramid& sp, int nLevels, int level)
{
    if (level < 0 || level >= nLevels) return;
    resizeSourcePyramid(sp, nLevels);
    if (!sp.nx[level].empty()) return;
    ensureCachedGrayLevel(sp, nLevels, level);
    computeGradLevel(sp.gray[level], sp.nx[level], sp.ny[level], sp.mag[level]);
}

static void ensureCachedPyramidLevel(SourcePyramid& sp, int nLevels, int level)
{
    ensureCachedGradLevel(sp, nLevels, level);
}

// When nLevels>=2, L0 fine search uses local ROI Sobel — skip full-image L0 grad until needed.
// P18 belt pyramid=2: coarse+fine at L0 only — never materialize L1 grad (Halcon §2.3 find NumLevels).
static int gradBuildStartLevel(int nLevels, int throughLevel, bool l0OnlyMode = false)
{
    if (l0OnlyMode && throughLevel == 0) return 0;
    return (throughLevel > 0 && nLevels > 1) ? 1 : 0;
}

static bool pyramidGradsReady(const SourcePyramid& sp, int nLevels, int throughLevel,
                              bool l0OnlyMode = false)
{
    const int start = gradBuildStartLevel(nLevels, throughLevel, l0OnlyMode);
    if (sp.nx.size() != (size_t)nLevels) return false;
    for (int l = start; l <= throughLevel && l < nLevels; ++l)
        if (sp.nx[l].empty()) return false;
    return true;
}

void computeGradLevel(const cv::Mat& gray, cv::Mat& nx, cv::Mat& ny, cv::Mat& mag) {
    cv::Mat gx, gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);
    nx.create(gray.size(), CV_32F);
    ny.create(gray.size(), CV_32F);
    const int H = gray.rows, W = gray.cols;
#pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        const float* gxr = gx.ptr<float>(y);
        const float* gyr = gy.ptr<float>(y);
        const float* mr  = mag.ptr<float>(y);
        float* nxr = nx.ptr<float>(y);
        float* nyr = ny.ptr<float>(y);
        for (int x = 0; x < W; ++x) {
            float m = mr[x];
            if (m < 1e-3f) { nxr[x] = 0.f; nyr[x] = 0.f; }
            else {
                float inv = 1.0f / m;
                nxr[x] = gxr[x] * inv;
                nyr[x] = gyr[x] * inv;
            }
        }
    }
}

// Halcon create_shape_model Optimization: fewer model points on coarse pyramid levels (§2.2.2).
static int pyramidFeatureBudget(int numFeatures, int level)
{
    int nf = std::max(16, numFeatures >> level);
    if (level > 0)
        nf = std::max(16, (nf * 2) / 3);
    return nf;
}

static void sampleRowMaxMag(const cv::Mat& mag, std::vector<float>& rowMax)
{
    if (mag.empty()) { rowMax.clear(); return; }
    rowMax.resize((size_t)mag.rows);
    const int sampleX = std::max(8, mag.cols / 128);
    const int H = mag.rows;
    auto scanRow = [&](int ry) {
        float rm = 0.f;
        const float* mr = mag.ptr<float>(ry);
        for (int x = 0; x < mag.cols; x += sampleX)
            rm = std::max(rm, mr[x]);
        rowMax[(size_t)ry] = rm;
    };
    if (H > 512) {
#pragma omp parallel for schedule(static)
        for (int ry = 0; ry < H; ++ry)
            scanRow(ry);
    } else {
        for (int ry = 0; ry < H; ++ry)
            scanRow(ry);
    }
}

struct CoarseModelCtx {
    int mi;
    const ShapeModelLevel* ml;
    const ShapeEdgeFeature* feats;
    const int* ox;
    const int* oy;
    int n, yMax, xMax;
    float thr;
};

void ensureSourcePyramidLevel(SourcePyramid& sp, const cv::Mat& src8u,
                              int nLevels, int level)
{
    if (level < 0 || level >= nLevels) return;
    if (sp.nx.size() != (size_t)nLevels) {
        sp.gray.assign(nLevels, cv::Mat());
        sp.nx.assign(nLevels, cv::Mat());
        sp.ny.assign(nLevels, cv::Mat());
        sp.mag.assign(nLevels, cv::Mat());
    }
    if (!sp.nx[level].empty()) return;

    if (level == 0) {
        sp.gray[0] = src8u;
    } else {
        if (sp.gray[level - 1].empty())
            ensureSourcePyramidLevel(sp, src8u, nLevels, level - 1);
        cv::Mat down;
        cv::pyrDown(sp.gray[level - 1], down);
        sp.gray[level] = down;
    }
    computeGradLevel(sp.gray[level], sp.nx[level], sp.ny[level], sp.mag[level]);
}

void buildSourcePyramid(const cv::Mat& src8u, int numLevels, SourcePyramid& out) {
    for (int l = 0; l < numLevels; ++l)
        ensureSourcePyramidLevel(out, src8u, numLevels, l);
}

void buildSourcePyramidThrough(SourcePyramid& out, const cv::Mat& src8u,
                               int nLevels, int throughLevel)
{
    for (int l = 0; l <= throughLevel && l < nLevels; ++l)
        ensureSourcePyramidLevel(out, src8u, nLevels, l);
}

// --- scoring -----------------------------------------------------------------
//
// Halcon find_shape_model Score: (1/N) * sum of per-point alignment contributions.
// use_polarity: contrib = max(0, dot(model_grad, scene_grad)); scene |grad| >= MinContrast.
// Coarse search uses alignment; reported Score uses scoreAtHalcon() below.

inline void accumulateAlignment(float d, ShapeMatchMetric metric, float& accPos, float& accNeg) {
    switch (metric) {
    case ShapeMatchMetric::UsePolarity:
        accPos += std::max(0.f, d);
        break;
    case ShapeMatchMetric::IgnoreLocalPolarity:
        accPos += std::fabs(d);
        break;
    case ShapeMatchMetric::IgnoreGlobalPolarity:
        accPos += d;
        accNeg += -d;
        break;
    }
}

// Bilinear sample of the (unit gradient field, magnitude) at a floating-point
// location. Returns false (caller treats as flat) if the 2x2 footprint would
// read out of bounds.
inline bool sampleGradBilinear(
    const cv::Mat& nx, const cv::Mat& ny, const cv::Mat& mag,
    float fx, float fy,
    float& snx, float& sny, float& smag)
{
    const int x0 = (int)std::floor(fx);
    const int y0 = (int)std::floor(fy);
    if (x0 < 0 || y0 < 0 || x0 + 1 >= nx.cols || y0 + 1 >= nx.rows) return false;

    const float ax = fx - (float)x0;
    const float ay = fy - (float)y0;
    const float w00 = (1.f - ax) * (1.f - ay);
    const float w10 = ax * (1.f - ay);
    const float w01 = (1.f - ax) * ay;
    const float w11 = ax * ay;

    const int sNx = (int)(nx.step / sizeof(float));
    const int sMg = (int)(mag.step / sizeof(float));
    const float* nxp = nx.ptr<float>(0) + y0 * sNx + x0;
    const float* nyp = ny.ptr<float>(0) + y0 * sNx + x0;
    const float* mgp = mag.ptr<float>(0) + y0 * sMg + x0;

    snx  = w00 * nxp[0]   + w10 * nxp[1]   + w01 * nxp[sNx]   + w11 * nxp[sNx + 1];
    sny  = w00 * nyp[0]   + w10 * nyp[1]   + w01 * nyp[sNx]   + w11 * nyp[sNx + 1];
    smag = w00 * mgp[0]   + w10 * mgp[1]   + w01 * mgp[sMg]   + w11 * mgp[sMg + 1];
    return true;
}

inline float metricScore(float accPos, float accNeg, int n, ShapeMatchMetric metric) {
    if (n <= 0) return 0.f;
    if (metric == ShapeMatchMetric::IgnoreGlobalPolarity)
        return std::max(accPos, accNeg) / (float)n;
    return accPos / (float)n;
}

inline float partialMetricScore(float accPos, float accNeg, ShapeMatchMetric metric) {
    if (metric == ShapeMatchMetric::IgnoreGlobalPolarity)
        return std::max(accPos, accNeg);
    return accPos;
}

inline float levelMinScore(float userScore, int /*level*/) {
    // Halcon: same MinScore at every pyramid level (find_shape_model §MinScore).
    return userScore;
}

// Halcon greediness: interpolate pessimistic (safe) and optimistic (fast) partial-sum bounds.
// partialSum = sum of per-feature contributions (each in [0,1] for use_polarity).
inline float halconGreedinessMinSum(float minScoreAvg, int j, int n, float greediness) {
    const float g = std::max(0.f, std::min(1.f, greediness));
    const float safe = minScoreAvg * (float)n - (float)(n - j);
    const float fast = minScoreAvg * (float)j;
    return (1.f - g) * safe + g * fast;
}

inline bool halconPartialKill(
    float accPos, float accNeg, int j, int n,
    float minScoreAvg, float greediness, ShapeMatchMetric metric)
{
    const float partial = partialMetricScore(accPos, accNeg, metric);
    return partial < halconGreedinessMinSum(minScoreAvg, j, n, greediness);
}

inline float scoreAtAlignment(
    const cv::Mat& nx, const cv::Mat& ny, const cv::Mat& mag,
    float x, float y,
    const ShapeEdgeFeature* features, int n,
    float magFloor, ShapeMatchMetric metric)
{
    float accPos = 0.f, accNeg = 0.f;
    for (int i = 0; i < n; ++i) {
        float snx, sny, smag;
        if (!sampleGradBilinear(nx, ny, mag,
                                x + features[i].dx, y + features[i].dy,
                                snx, sny, smag))
            continue;
        if (smag < magFloor) continue;
        float d = features[i].gx * snx + features[i].gy * sny;
        accumulateAlignment(d, metric, accPos, accNeg);
    }
    return metricScore(accPos, accNeg, n, metric);
}

// Halcon find_shape_model Score (use_polarity): mean positive dot product over N model points
// with scene contrast >= MinContrast. Points below contrast or outside the image contribute 0.
inline float scoreAtHalcon(
    const cv::Mat& nx, const cv::Mat& ny, const cv::Mat& mag,
    float x, float y,
    const ShapeEdgeFeature* features, int n,
    float magFloor, ShapeMatchMetric metric)
{
    return scoreAtAlignment(nx, ny, mag, x, y, features, n, magFloor, metric);
}

// P42 Halcon partial visibility: score over contrast-visible points only (large template borderline).
inline float scoreAtHalconVisible(
    const cv::Mat& nx, const cv::Mat& ny, const cv::Mat& mag,
    float x, float y,
    const ShapeEdgeFeature* features, int n,
    float magFloor, ShapeMatchMetric metric)
{
    float accPos = 0.f, accNeg = 0.f;
    int visible = 0;
    for (int i = 0; i < n; ++i) {
        float snx, sny, smag;
        if (!sampleGradBilinear(nx, ny, mag,
                x + features[i].dx, y + features[i].dy, snx, sny, smag))
            continue;
        if (smag < magFloor) continue;
        ++visible;
        float d = features[i].gx * snx + features[i].gy * sny;
        accumulateAlignment(d, metric, accPos, accNeg);
    }
    if (visible < 4) return 0.f;
    return metricScore(accPos, accNeg, visible, metric);
}

inline float scoreAtHalconBest(
    const cv::Mat& nx, const cv::Mat& ny, const cv::Mat& mag,
    float x, float y,
    const ShapeEdgeFeature* features, int n,
    float magFloor, ShapeMatchMetric metric, bool useVisibleDenom)
{
    const float sStd = scoreAtHalcon(nx, ny, mag, x, y, features, n, magFloor, metric);
    if (!useVisibleDenom) return sStd;
    const float sVis = scoreAtHalconVisible(nx, ny, mag, x, y, features, n, magFloor, metric);
    return std::max(sStd, sVis);
}

// Fine/coarse pose scoring with Halcon greediness early-out (ordered feature eval).
inline float scoreAtAlignmentGreediness(
    const cv::Mat& nx, const cv::Mat& ny, const cv::Mat& mag,
    float x, float y,
    const ShapeEdgeFeature* features, int n,
    float magFloor, ShapeMatchMetric metric,
    float minScoreAvg, float greediness)
{
    float accPos = 0.f, accNeg = 0.f;
    for (int i = 0; i < n; ++i) {
        float snx, sny, smag;
        if (!sampleGradBilinear(nx, ny, mag,
                                x + features[i].dx, y + features[i].dy,
                                snx, sny, smag))
            continue;
        if (smag < magFloor) continue;
        float d = features[i].gx * snx + features[i].gy * sny;
        accumulateAlignment(d, metric, accPos, accNeg);
        if (halconPartialKill(accPos, accNeg, i + 1, n, minScoreAvg, greediness, metric))
            return -1.f;
    }
    return metricScore(accPos, accNeg, n, metric);
}

// (Halcon parity: no extra borderline re-score gate — find_shape_model uses MinScore only.)

// Coarse-level alignment scoring with SIMD early-out and Halcon greediness.
// featStride>1: PatMax coarse granularity — evaluate every Nth feature (§Feature Size doc).
inline bool coarseScoreAccumulate(
    int x, int y, int n,
    const ShapeEdgeFeature* feats,
    const int* ox, const int* oy,
    const float* nxBase, const float* nyBase, const float* mgBase,
    int stepNx, int stepMg,
    float magFloor, ShapeMatchMetric metric,
    float minScoreAvg, float greediness,
    float& accPos, float& accNeg,
    int featStride = 1,
    int imgCols = 0, int imgRows = 0)
{
    accPos = accNeg = 0.f;
    if (n <= 0) return true;
    if (featStride < 1) featStride = 1;
    const int nEff = (n + featStride - 1) / featStride;
    int i = 0;
    const auto inBounds = [&](int sx, int sy) -> bool {
        if (imgCols <= 0 || imgRows <= 0) return true;
        return (unsigned)sx < (unsigned)imgCols && (unsigned)sy < (unsigned)imgRows;
    };

    if (featStride == 1) {
    auto processChunk = [&](int count) {
        for (int k = 0; k < count; ++k) {
            int sx = x + ox[i + k];
            int sy = y + oy[i + k];
            if (!inBounds(sx, sy)) continue;
            float m = mgBase[sy * stepMg + sx];
            if (m >= magFloor) {
                float snx = nxBase[sy * stepNx + sx];
                float sny = nyBase[sy * stepNx + sx];
                float d = feats[i + k].gx * snx + feats[i + k].gy * sny;
                accumulateAlignment(d, metric, accPos, accNeg);
            }
        }
    };

#ifdef SHAPE_HAS_AVX2
    if (metric == ShapeMatchMetric::UsePolarity ||
        metric == ShapeMatchMetric::IgnoreLocalPolarity)
    {
        const __m256 signMask = _mm256_set1_ps(-0.f);
        const __m256 zero = _mm256_setzero_ps();
        for (; i + 7 < n; i += 8) {
            alignas(32) float snx[8], sny[8], sm[8];
            for (int k = 0; k < 8; ++k) {
                int sx = x + ox[i + k];
                int sy = y + oy[i + k];
                if (!inBounds(sx, sy)) { sm[k] = 0.f; snx[k] = 0.f; sny[k] = 0.f; continue; }
                sm[k]  = mgBase[sy * stepMg + sx];
                snx[k] = nxBase[sy * stepNx + sx];
                sny[k] = nyBase[sy * stepNx + sx];
            }
            __m256 vSm  = _mm256_loadu_ps(sm);
            __m256 vFlr = _mm256_set1_ps(magFloor);
            __m256 vMask = _mm256_cmp_ps(vSm, vFlr, _CMP_GE_OQ);

            __m256 vGx = _mm256_set_ps(
                feats[i + 7].gx, feats[i + 6].gx, feats[i + 5].gx, feats[i + 4].gx,
                feats[i + 3].gx, feats[i + 2].gx, feats[i + 1].gx, feats[i].gx);
            __m256 vGy = _mm256_set_ps(
                feats[i + 7].gy, feats[i + 6].gy, feats[i + 5].gy, feats[i + 4].gy,
                feats[i + 3].gy, feats[i + 2].gy, feats[i + 1].gy, feats[i].gy);
            __m256 vNx = _mm256_loadu_ps(snx);
            __m256 vNy = _mm256_loadu_ps(sny);
            __m256 vDot = _mm256_add_ps(_mm256_mul_ps(vGx, vNx), _mm256_mul_ps(vGy, vNy));

            if (metric == ShapeMatchMetric::IgnoreLocalPolarity)
                vDot = _mm256_andnot_ps(signMask, vDot);
            else if (metric == ShapeMatchMetric::UsePolarity)
                vDot = _mm256_max_ps(vDot, zero);
            vDot = _mm256_and_ps(vDot, vMask);

            alignas(32) float dots[8];
            _mm256_storeu_ps(dots, vDot);
            for (int k = 0; k < 8; ++k)
                accPos += dots[k];

            if (halconPartialKill(accPos, accNeg, i + 8, n, minScoreAvg, greediness, metric))
                return false;
        }
    }
#endif

#ifdef SHAPE_HAS_SSE2
    if (metric == ShapeMatchMetric::UsePolarity ||
        metric == ShapeMatchMetric::IgnoreLocalPolarity)
    {
        for (; i + 3 < n; i += 4) {
            processChunk(4);
            if (halconPartialKill(accPos, accNeg, i + 4, n, minScoreAvg, greediness, metric))
                return false;
        }
    }
#endif

    for (; i < n; ++i) {
        int sx = x + ox[i];
        int sy = y + oy[i];
        if (!inBounds(sx, sy)) {
            if (halconPartialKill(accPos, accNeg, i + 1, n, minScoreAvg, greediness, metric))
                return false;
            continue;
        }
        float m = mgBase[sy * stepMg + sx];
        if (m >= magFloor) {
            float snx = nxBase[sy * stepNx + sx];
            float sny = nyBase[sy * stepNx + sx];
            float d = feats[i].gx * snx + feats[i].gy * sny;
            accumulateAlignment(d, metric, accPos, accNeg);
        }
        if (halconPartialKill(accPos, accNeg, i + 1, n, minScoreAvg, greediness, metric))
            return false;
    }
    return true;
    } // featStride==1

    int j = 0;
    for (i = 0; i < n; i += featStride, ++j) {
        int sx = x + ox[i];
        int sy = y + oy[i];
        if (!inBounds(sx, sy)) {
            if (halconPartialKill(accPos, accNeg, j + 1, nEff, minScoreAvg, greediness, metric))
                return false;
            continue;
        }
        float m = mgBase[sy * stepMg + sx];
        if (m >= magFloor) {
            float snx = nxBase[sy * stepNx + sx];
            float sny = nyBase[sy * stepNx + sx];
            float d = feats[i].gx * snx + feats[i].gy * sny;
            accumulateAlignment(d, metric, accPos, accNeg);
        }
        if (halconPartialKill(accPos, accNeg, j + 1, nEff, minScoreAvg, greediness, metric))
            return false;
    }
    return true;
}

cv::Mat scaleTemplate(const cv::Mat& gray, const cv::Mat& mask, float scale,
                      cv::Mat& scaledMaskOut) {
    if (std::abs(scale - 1.0f) < 1e-4f) {
        scaledMaskOut = mask;
        return gray;
    }
    cv::Size newSz(
        std::max(1, (int)std::lround(gray.cols * scale)),
        std::max(1, (int)std::lround(gray.rows * scale)));
    cv::Mat scaledGray, scaledMask;
    cv::resize(gray, scaledGray, newSz, 0, 0, cv::INTER_LINEAR);
    if (!mask.empty())
        cv::resize(mask, scaledMask, newSz, 0, 0, cv::INTER_NEAREST);
    else
        scaledMask = cv::Mat(newSz, CV_8UC1, cv::Scalar(255));
    scaledMaskOut = scaledMask;
    return scaledGray;
}

cv::Mat anisoScaleTemplate(const cv::Mat& gray, const cv::Mat& mask,
                           float scaleR, float scaleC, cv::Mat& scaledMaskOut) {
    if (std::abs(scaleR - 1.0f) < 1e-4f && std::abs(scaleC - 1.0f) < 1e-4f) {
        scaledMaskOut = mask;
        return gray;
    }
    cv::Size newSz(
        std::max(1, (int)std::lround(gray.cols * scaleC)),
        std::max(1, (int)std::lround(gray.rows * scaleR)));
    cv::Mat scaledGray, scaledMask;
    cv::resize(gray, scaledGray, newSz, 0, 0, cv::INTER_LINEAR);
    if (!mask.empty())
        cv::resize(mask, scaledMask, newSz, 0, 0, cv::INTER_NEAREST);
    else
        scaledMask = cv::Mat(newSz, CV_8UC1, cv::Scalar(255));
    scaledMaskOut = scaledMask;
    return scaledGray;
}

double computeAutoAngleStep(int templateWidth, int templateHeight) {
    const int maxDim = std::max(templateWidth, templateHeight);
    if (maxDim <= 0) return 1.0;
    double step = std::atan(2.0 / (double)maxDim) * 180.0 / CV_PI;
    if (step < 0.1) step = 0.1;
    return step;
}

int computeAutoNumPyramidLevels(int templateWidth, int templateHeight) {
    const int minDim = std::min(templateWidth, templateHeight);
    if (minDim <= 16) return 1;
    int levels = (int)std::floor(std::log2(minDim / 16.0));
    return std::max(1, std::min(6, levels));
}

// Resolve trained pyramid depth: cap by template size.
// Auto mode (requestedLevels==0): geometry-optimal depth, >=2 when allowed.
// Explicit NumLevels (Halcon sync): minimum depth; may deepen when geometry allows if
// self-match at train time passes (Cognex/Halcon speed — same find semantics).
int resolveActualNumLevels(int requestedLevels, int templateWidth, int templateHeight,
                           bool allowGeometryBoost = true) {
    const int autoLevels = computeAutoNumPyramidLevels(templateWidth, templateHeight);
    const bool explicitRequest = requestedLevels > kAutoNumPyramidLevels;
    // P16 Halcon §2.3: wide belt templates train/search with 2 pyramid levels.
    const bool beltWide = templateWidth >= templateHeight * 3 && templateWidth >= 120
        && templateHeight >= 20;

    int allowed = 1;
    const int minDim = std::min(templateWidth, templateHeight);
    while ((minDim >> allowed) >= 24 && allowed < 6) ++allowed;
    if (beltWide)
        allowed = std::max(allowed, 2);

    if (!explicitRequest) {
        int actual = std::max(1, std::min(autoLevels, allowed));
        if (allowed >= 2)
            actual = std::max(actual, 2);
        return actual;
    }

    int actual = std::max(1, std::min(requestedLevels, allowed));
    if (!allowGeometryBoost || requestedLevels <= 1) {
        if (beltWide && allowGeometryBoost && allowed >= 2)
            return 2;
        return actual;
    }

    const int geo = std::min(autoLevels, allowed);
    if (geo > actual)
        return std::min(geo, actual + 1);
    return actual;
}

void buildMaskPyramid(const cv::Mat& mask8u, int numLevels,
                      std::vector<cv::Mat>& out)
{
    out.clear();
    if (mask8u.empty()) return;
    cv::Mat cur = mask8u;
    out.resize(numLevels);
    for (int l = 0; l < numLevels; ++l) {
        if (l > 0) {
            cv::Mat down;
            cv::pyrDown(cur, down);
            cv::threshold(down, down, 127, 255, cv::THRESH_BINARY);
            cur = down;
        } else {
            cur = mask8u;
        }
        out[l] = cur;
    }
}

// Halcon reduce_domain equivalent: crop template + mask to mask bounding box so
// pyramid/feature extraction uses the effective ROI, not the full canvas.
bool cropGrayMaskToRoi(cv::Mat& gray, cv::Mat& mask,
                       double& originX, double& originY, bool originSet)
{
    if (mask.empty()) return true;
    cv::Mat bin = mask;
    if (bin.type() != CV_8UC1)
        cv::cvtColor(bin, bin, cv::COLOR_BGR2GRAY);
    cv::Rect roi = cv::boundingRect(bin);
    if (roi.width <= 0 || roi.height <= 0) return false;
    roi &= cv::Rect(0, 0, gray.cols, gray.rows);
    if (roi.width <= 0 || roi.height <= 0) return false;

    gray = gray(roi).clone();
    mask = mask(roi).clone();
    if (originSet) {
        // Halcon reports (0,0) for auto origin — use cropped-image center, not literal top-left.
        if (std::abs(originX) < 1e-6 && std::abs(originY) < 1e-6) {
            originX = gray.cols * 0.5;
            originY = gray.rows * 0.5;
        } else {
            originX -= roi.x;
            originY -= roi.y;
        }
    } else {
        originX = gray.cols * 0.5;
        originY = gray.rows * 0.5;
    }
    return true;
}

// Coarse pyramid: gradient magnitudes are lower (blur). Scale MinContrast floor per level.
inline float coarseMagFloor(float baseMagFloor, int nLevels, int level)
{
    if (nLevels <= 1 || level <= 0) return baseMagFloor;
    const float scale = std::pow(0.55f, (float)(nLevels - 1 - level));
    return std::max(1.f, baseMagFloor * scale);
}

// PatMax/Halcon coarse stage: Accept threshold is below fine MinScore so weak-but-valid
// coarse peaks still enter refinement (MVTec coarse-to-fine; Halcon §find_shape_model).
inline float coarseAcceptThreshold(float userScore01, float greediness, int nLevels)
{
    if (nLevels <= 1)
        return std::max(0.06f, userScore01 * (0.35f + 0.25f * greediness));
    const float ratio = std::max(0.30f, 0.55f - 0.25f * greediness);
    return std::max(0.06f, userScore01 * ratio);
}

// PatMax Accept score (coarse filter) — tied to user MinScore, not a fixed low floor.
inline float patmaxCoarseAcceptScore(float userMinScore, float greediness, int nLevels)
{
    const float ratio = (nLevels <= 1)
        ? std::max(0.38f, 0.65f - 0.22f * greediness)
        : std::max(0.30f, 0.58f - 0.25f * greediness);
    return std::max(0.10f, userMinScore * ratio);
}

// Halcon NumMatches: limit candidates entering fine refine (§2.4.4 Solution Guide II-B).
inline int halconFinePoolCap(int maxTargets, float greediness, bool beltLarge, int nLevels,
                             bool smallScene)
{
    if (maxTargets <= 0) return 8;
    // Belt large scene: same pool depth regardless of NumMatches (coarse rank ≠ fine rank).
    if (beltLarge && greediness >= 0.9f && !smallScene)
        return 48;
    // NumMatches=1 + high greediness: small pool, but large scenes need depth for
    // coarse-rank≠fine-rank (154617 / 130530 borderline at mt=1).
    if (maxTargets == 1 && greediness >= 0.9f) {
        if (smallScene) return 4;
        return beltLarge ? 48 : 24;
    }
    int cap = std::max(maxTargets * (greediness >= 0.9f ? 6 : 12), 8);
    if (nLevels > 1)
        cap = std::max(cap, maxTargets * (greediness >= 0.85f ? 4 : 10));
    else if (beltLarge)
        cap = std::max(cap, std::max(maxTargets * 8, 12));
    if (smallScene && maxTargets == 1)
        cap = std::min(cap, greediness >= 0.9f ? 4 : 8);
    return cap;
}

} // namespace

ShapeModelParams determineShapeModelParams(cv::InputArray templateImg,
                                           cv::InputArray mask)
{
    ShapeModelParams p;
    p.numPyramidLevels = kAutoNumPyramidLevels;
    p.angleStep = 1.0;
    p.minContrast = 10.f;
    p.numFeatures = 128;

    cv::Mat gray = ensureGray8u(templateImg.getMat());
    if (gray.empty()) return p;

    cv::Mat useMask;
    if (!mask.empty()) {
        cv::Mat m = mask.getMat();
        if (m.size() == gray.size() && m.type() == CV_8UC1)
            useMask = m;
    }

    double ox = 0, oy = 0;
    if (!useMask.empty())
        cropGrayMaskToRoi(gray, useMask, ox, oy, false);

    p.numPyramidLevels = resolveActualNumLevels(p.numPyramidLevels, gray.cols, gray.rows);
    p.angleStep = computeAutoAngleStep(gray.cols, gray.rows);

    const double area = (double)gray.cols * gray.rows;

    cv::Mat gx, gy, mag;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);

    std::vector<float> mags;
    mags.reserve(4096);
    for (int y = 1; y < mag.rows - 1; ++y) {
        const float* row = mag.ptr<float>(y);
        const uchar* mk = useMask.empty() ? nullptr : useMask.ptr<uchar>(y);
        for (int x = 1; x < mag.cols - 1; ++x) {
            if (mk && mk[x] == 0) continue;
            float v = row[x];
            if (v > 2.f) mags.push_back(v);
        }
    }

    if (!mags.empty()) {
        std::sort(mags.begin(), mags.end());
        p.minContrast = mags[mags.size() / 4];
        p.minContrast = std::max(5.f, std::min(45.f, p.minContrast));
        // Flat background (polygon-on-canvas): raise contrast so weak BG edges are ignored.
        if (area >= 80000.0) {
            const float p25 = mags[mags.size() / 4];
            const float p90 = mags[(size_t)(mags.size() * 0.9)];
            if (p90 < p25 * 0.55f)
                p.minContrast = std::max(p.minContrast, std::min(45.f, p25 * 1.15f));
        }
    }

    p.numFeatures = (int)std::lround(std::sqrt(area) * 1.15);
    p.numFeatures = std::max(32, std::min(128, p.numFeatures));

    const int maxDim = std::max(gray.cols, gray.rows);
    const int minDim = std::min(gray.cols, gray.rows);
    const float aspect = (float)maxDim / (float)std::max(1, minDim);
    // Elongated belt/crop templates: softer MinContrast floor so scene edges at L0
    // are not over-filtered (Halcon auto-contrast is lower on thin horizontal ROIs).
    const float smallTplFloor = (aspect >= 3.f) ? 18.f : 28.f;
    if (maxDim < 128)
        p.minContrast = std::max(p.minContrast, aspect >= 3.f ? 22.f : 40.f);
    else if (maxDim < 256)
        p.minContrast = std::max(p.minContrast, smallTplFloor);

    return p;
}

// ============================================================================
// ShapeBasedMatcher implementation
// ============================================================================

cv::Ptr<ShapeBasedMatcher> ShapeBasedMatcher::create() {
    return cv::makePtr<ShapeBasedMatcherImpl>();
}

void ShapeBasedMatcherImpl::clearSceneGradientCache()
{
    m_scW = m_scH = m_scLevels = 0;
    m_scFp = 0;
    m_scBuiltThrough = -1;
    m_scenePyramid = SourcePyramid();
}

void ShapeBasedMatcherImpl::ensureScenePyramid(const cv::Mat& srcGray, int nLevels,
                                               int throughLevel, SourcePyramid& out,
                                               bool l0OnlyMode)
{
    if (nLevels <= 0) return;
    throughLevel = std::max(0, std::min(throughLevel, nLevels - 1));
    const std::uint64_t fp = sceneFingerprint(srcGray);

    // Per-instance fast path (same scene, repeated find on one matcher — no mutex).
    if (m_scFp == fp && m_scW == srcGray.cols && m_scH == srcGray.rows
        && m_scLevels == nLevels
        && pyramidGradsReady(m_scenePyramid, nLevels, throughLevel, l0OnlyMode)) {
        out.gray = m_scenePyramid.gray;
        out.nx = m_scenePyramid.nx;
        out.ny = m_scenePyramid.ny;
        out.mag = m_scenePyramid.mag;
        return;
    }

    std::lock_guard<std::mutex> lock(g_sceneCache.mu);

    CachedSceneEntry* entry = nullptr;
    auto it = g_sceneCache.entries.find(fp);
    if (it != g_sceneCache.entries.end()
        && it->second.w == srcGray.cols && it->second.h == srcGray.rows
        && it->second.levels == nLevels) {
        entry = &it->second;
        g_sceneCache.lru.remove(fp);
        g_sceneCache.lru.push_front(fp);
    } else {
        if (it != g_sceneCache.entries.end())
            g_sceneCache.entries.erase(it);
        while (g_sceneCache.entries.size() >= GlobalScenePyramidCache::kMaxEntries
               && !g_sceneCache.lru.empty()) {
            g_sceneCache.entries.erase(g_sceneCache.lru.back());
            g_sceneCache.lru.pop_back();
        }
        CachedSceneEntry neu;
        neu.w = srcGray.cols;
        neu.h = srcGray.rows;
        neu.levels = nLevels;
        neu.fp = fp;
        neu.builtThrough = -1;
        neu.pyramid.gray.assign(nLevels, cv::Mat());
        neu.pyramid.nx.assign(nLevels, cv::Mat());
        neu.pyramid.ny.assign(nLevels, cv::Mat());
        neu.pyramid.mag.assign(nLevels, cv::Mat());
        neu.pyramid.gray[0] = srcGray.clone();
        auto ins = g_sceneCache.entries.emplace(fp, std::move(neu));
        entry = &ins.first->second;
        g_sceneCache.lru.push_front(fp);
    }

    const int grayThrough = (l0OnlyMode && throughLevel == 0) ? 0 : throughLevel;
    for (int l = 0; l <= grayThrough && l < nLevels; ++l)
        ensureCachedGrayLevel(entry->pyramid, nLevels, l);
    const int gradStart = gradBuildStartLevel(nLevels, throughLevel, l0OnlyMode);
    for (int l = gradStart; l <= throughLevel && l < nLevels; ++l)
        ensureCachedGradLevel(entry->pyramid, nLevels, l);
    entry->builtThrough = std::max(entry->builtThrough, throughLevel);

    m_scW = entry->w;
    m_scH = entry->h;
    m_scFp = entry->fp;
    m_scLevels = entry->levels;
    m_scBuiltThrough = entry->builtThrough;
    m_scenePyramid = entry->pyramid;

    out.gray = entry->pyramid.gray;
    out.nx = entry->pyramid.nx;
    out.ny = entry->pyramid.ny;
    out.mag = entry->pyramid.mag;
}

ShapeBasedMatcherImpl::ShapeBasedMatcherImpl()
    : m_originX(0.0)
    , m_originY(0.0)
    , m_originCustom(false)
    , m_minAngle(0.0)
    , m_maxAngle(0.0)
    , m_angleStep(1.0)
    , m_effectiveAngleStep(1.0)
    , m_minScale(1.0)
    , m_maxScale(1.0)
    , m_scaleStep(0.0)
    , m_minScaleR(1.0)
    , m_maxScaleR(1.0)
    , m_scaleStepR(0.0)
    , m_minScaleC(1.0)
    , m_maxScaleC(1.0)
    , m_scaleStepC(0.0)
    , m_anisoScaleEnabled(false)
    , m_numPyramidLevels(4)
    , m_greediness(0.9)
    , m_maxOverlap(0.5)
    , m_subPixelMode(ShapeSubPixelMode::LeastSquares)
    , m_findTimeoutMs(0)
    , m_useIcpRefine(false)
    , m_icpMaxIterations(5)
    , m_minContrast(10.f)
    , m_numFeatures(128)
    , m_metric(ShapeMatchMetric::UsePolarity)
    , m_metricUserSet(false)
    , m_contourRefW(0)
    , m_contourRefH(0)
    , m_cropTemplateToMask(true)
    , m_positionBiasX(0.0)
    , m_positionBiasY(0.0)
    , m_findHintCol(0.0)
    , m_findHintRow(0.0)
    , m_findHintScore(-1.0)
    , m_hasFindHint(false)
    , m_trackLastFind(false)
    , m_findLowestPyramidLevel(0)
    , m_actualNumLevels(0)
    , m_templateRefW(0)
    , m_templateRefH(0)
    , m_isTrained(false)
{
}

ShapeBasedMatcherImpl::~ShapeBasedMatcherImpl() {}

// ---- template / origin management ----

bool ShapeBasedMatcherImpl::prepareMask(cv::InputArray templateImg, cv::InputArray mask,
                                        cv::Mat& outMask) const
{
    cv::Mat t = templateImg.getMat();
    if (t.empty()) return false;
    if (mask.empty()) {
        if (t.channels() == 4) {
            std::vector<cv::Mat> ch;
            cv::split(t, ch);
            cv::threshold(ch[3], outMask, 127, 255, cv::THRESH_BINARY);
        } else {
            outMask = cv::Mat(t.size(), CV_8UC1, cv::Scalar(255));
        }
    } else {
        cv::Mat m = mask.getMat();
        if (m.size() != t.size() || m.type() != CV_8UC1) return false;
        outMask = m.clone();
    }
    return true;
}

bool ShapeBasedMatcherImpl::setTemplate(cv::InputArray templateImg, cv::InputArray mask) {
    cv::Mat t = templateImg.getMat();
    if (t.empty()) return false;
    m_pending.clear();
    m_templateMeta.clear();
    m_models.clear();
    m_isTrained = false;

    PendingTemplate pt;
    pt.name = "";
    pt.fromContour = false;
    pt.contourRefW = pt.contourRefH = 0;
    pt.image = t.clone();
    if (!prepareMask(templateImg, mask, pt.mask)) return false;
    pt.originX = m_originX;
    pt.originY = m_originY;
    pt.originSet = m_originCustom;
    m_pending.push_back(std::move(pt));

    m_templateImg = m_pending.back().image.clone();
    m_mask = m_pending.back().mask.clone();
    return true;
}

void ShapeBasedMatcherImpl::clearTemplates() {
    m_pending.clear();
    m_templateMeta.clear();
    m_models.clear();
    m_templateImg.release();
    m_mask.release();
    m_isTrained = false;
}

int ShapeBasedMatcherImpl::addTemplate(cv::InputArray templateImg, cv::InputArray mask,
                                       const std::string& name)
{
    cv::Mat t = templateImg.getMat();
    if (t.empty()) return -1;

    PendingTemplate pt;
    pt.name = name;
    pt.fromContour = false;
    pt.contourRefW = pt.contourRefH = 0;
    pt.image = t.clone();
    if (!prepareMask(templateImg, mask, pt.mask)) return -1;
    pt.originX = m_originX;
    pt.originY = m_originY;
    pt.originSet = m_originCustom;
    m_pending.push_back(std::move(pt));
    m_isTrained = false;
    m_models.clear();
    m_templateMeta.clear();
    return (int)m_pending.size() - 1;
}

void ShapeBasedMatcherImpl::setOrigin(double x, double y) {
    m_originX = x;
    m_originY = y;
    m_originCustom = true;
    if (!m_pending.empty()) {
        m_pending.back().originX = x;
        m_pending.back().originY = y;
        m_pending.back().originSet = true;
    }
}

double ShapeBasedMatcherImpl::getOriginX() const {
    if (!m_pending.empty() && m_pending.back().originSet)
        return m_pending.back().originX;
    if (m_originCustom) return m_originX;
    if (!m_templateImg.empty()) return m_templateImg.cols / 2.0;
    return 0.0;
}

double ShapeBasedMatcherImpl::getOriginY() const {
    if (!m_pending.empty() && m_pending.back().originSet)
        return m_pending.back().originY;
    if (m_originCustom) return m_originY;
    if (!m_templateImg.empty()) return m_templateImg.rows / 2.0;
    return 0.0;
}

void ShapeBasedMatcherImpl::resetOrigin() {
    m_originCustom = false;
    if (!m_pending.empty())
        m_pending.back().originSet = false;
}

void ShapeBasedMatcherImpl::setPositionBias(double x, double y) {
    m_positionBiasX = x;
    m_positionBiasY = y;
}

double ShapeBasedMatcherImpl::getPositionBiasX() const { return m_positionBiasX; }
double ShapeBasedMatcherImpl::getPositionBiasY() const { return m_positionBiasY; }

void ShapeBasedMatcherImpl::setFindHint(double col, double row) {
    m_findHintCol = col;
    m_findHintRow = row;
    m_hasFindHint = true;
}

void ShapeBasedMatcherImpl::clearFindHint() {
    m_hasFindHint = false;
    m_findHintScore = -1.0;
}

bool ShapeBasedMatcherImpl::hasFindHint() const { return m_hasFindHint; }

void ShapeBasedMatcherImpl::setTrackLastFind(bool enable) { m_trackLastFind = enable; }

bool ShapeBasedMatcherImpl::getTrackLastFind() const { return m_trackLastFind; }

void ShapeBasedMatcherImpl::setFindLowestPyramidLevel(int level) {
    m_findLowestPyramidLevel = std::max(0, level);
}

int ShapeBasedMatcherImpl::getFindLowestPyramidLevel() const {
    return m_findLowestPyramidLevel;
}

int ShapeBasedMatcherImpl::numModels() const { return (int)m_templateMeta.size(); }
int ShapeBasedMatcherImpl::numPoseModels() const { return (int)m_models.size(); }

// ---- trivial setters / getters ----

void ShapeBasedMatcherImpl::setAngleRange(double a, double b)   { m_minAngle = a; m_maxAngle = b; }
void ShapeBasedMatcherImpl::setAngleStep(double s)              { m_angleStep = s; }
double ShapeBasedMatcherImpl::getMinAngle() const               { return m_minAngle; }
double ShapeBasedMatcherImpl::getMaxAngle() const               { return m_maxAngle; }
double ShapeBasedMatcherImpl::getAngleStep() const              { return m_angleStep; }
double ShapeBasedMatcherImpl::getEffectiveAngleStep() const     { return m_effectiveAngleStep; }

void ShapeBasedMatcherImpl::setScaleRange(double a, double b, double s) {
    m_minScale = std::max(0.01, a);
    m_maxScale = std::max(m_minScale, b);
    m_scaleStep = s > 0.0 ? s : 0.0;
    m_anisoScaleEnabled = false;
}
double ShapeBasedMatcherImpl::getMinScale() const               { return m_minScale; }
double ShapeBasedMatcherImpl::getMaxScale() const               { return m_maxScale; }
double ShapeBasedMatcherImpl::getScaleStep() const              { return m_scaleStep; }

void ShapeBasedMatcherImpl::setAnisoScaleRange(double minR, double maxR, double stepR,
                                               double minC, double maxC, double stepC) {
    m_minScaleR = std::max(0.01, minR);
    m_maxScaleR = std::max(m_minScaleR, maxR);
    m_scaleStepR = stepR > 0.0 ? stepR : 0.0;
    m_minScaleC = std::max(0.01, minC);
    m_maxScaleC = std::max(m_minScaleC, maxC);
    m_scaleStepC = stepC > 0.0 ? stepC : 0.0;
    m_anisoScaleEnabled = true;
}
double ShapeBasedMatcherImpl::getMinScaleR() const  { return m_minScaleR; }
double ShapeBasedMatcherImpl::getMaxScaleR() const  { return m_maxScaleR; }
double ShapeBasedMatcherImpl::getScaleStepR() const { return m_scaleStepR; }
double ShapeBasedMatcherImpl::getMinScaleC() const  { return m_minScaleC; }
double ShapeBasedMatcherImpl::getMaxScaleC() const  { return m_maxScaleC; }
double ShapeBasedMatcherImpl::getScaleStepC() const { return m_scaleStepC; }
bool   ShapeBasedMatcherImpl::getAnisoScaleEnabled() const { return m_anisoScaleEnabled; }

void ShapeBasedMatcherImpl::setUseIcpRefine(bool enable) { m_useIcpRefine = enable; }
bool ShapeBasedMatcherImpl::getUseIcpRefine() const { return m_useIcpRefine; }
void ShapeBasedMatcherImpl::setIcpMaxIterations(int n) {
    m_icpMaxIterations = std::max(1, std::min(50, n));
}
int ShapeBasedMatcherImpl::getIcpMaxIterations() const { return m_icpMaxIterations; }

bool ShapeBasedMatcherImpl::parseContourInput(cv::InputArray points, cv::InputArray gradients,
                                              PendingTemplate& pt) const
{
    cv::Mat ptsMat = points.getMat();
    if (ptsMat.empty()) return false;

    cv::Mat pts2;
    if (ptsMat.channels() == 2 && ptsMat.cols == 1)
        ptsMat.reshape(2, ptsMat.rows).convertTo(pts2, CV_32F);
    else if (ptsMat.cols == 2 && ptsMat.rows >= 1)
        ptsMat.convertTo(pts2, CV_32F);
    else if (ptsMat.type() == CV_32FC2)
        ptsMat.reshape(1, ptsMat.rows).convertTo(pts2, CV_32F);
    else
        return false;

    const int n = pts2.rows;
    if (n < 3) return false;

    std::vector<cv::Point2f> xy(n);
    for (int i = 0; i < n; ++i) {
        xy[i].x = pts2.at<float>(i, 0);
        xy[i].y = pts2.at<float>(i, 1);
    }

    cv::Mat gradMat;
    bool haveGrad = !gradients.empty();
    if (haveGrad) {
        gradMat = gradients.getMat();
        cv::Mat g2;
        if (gradMat.channels() == 2 && gradMat.cols == 1)
            gradMat.reshape(2, gradMat.rows).convertTo(g2, CV_32F);
        else if (gradMat.cols == 2)
            gradMat.convertTo(g2, CV_32F);
        else
            haveGrad = false;
        if (haveGrad && g2.rows != n) haveGrad = false;
        else if (haveGrad) gradMat = g2;
    }

    float minX =  std::numeric_limits<float>::max();
    float minY =  std::numeric_limits<float>::max();
    float maxX = -std::numeric_limits<float>::max();
    float maxY = -std::numeric_limits<float>::max();
    for (const auto& p : xy) {
        minX = std::min(minX, p.x); minY = std::min(minY, p.y);
        maxX = std::max(maxX, p.x); maxY = std::max(maxY, p.y);
    }

    pt.contourRefW = std::max(1, (int)std::ceil(maxX - minX) + 1);
    pt.contourRefH = std::max(1, (int)std::ceil(maxY - minY) + 1);
    pt.contourFeatures.clear();
    pt.contourFeatures.reserve(n);

    for (int i = 0; i < n; ++i) {
        ShapeEdgeFeature f;
        f.dx = xy[i].x;
        f.dy = xy[i].y;
        if (haveGrad) {
            f.gx = gradMat.at<float>(i, 0);
            f.gy = gradMat.at<float>(i, 1);
            const float len = std::sqrt(f.gx * f.gx + f.gy * f.gy);
            if (len > 1e-6f) { f.gx /= len; f.gy /= len; }
            else { f.gx = 1.f; f.gy = 0.f; }
        } else {
            const int prev = (i > 0) ? i - 1 : n - 1;
            const int next = (i + 1 < n) ? i + 1 : 0;
            float tx = xy[next].x - xy[prev].x;
            float ty = xy[next].y - xy[prev].y;
            const float len = std::sqrt(tx * tx + ty * ty);
            if (len > 1e-6f) { f.gx = -ty / len; f.gy = tx / len; }
            else { f.gx = 1.f; f.gy = 0.f; }
        }
        pt.contourFeatures.push_back(f);
    }
    return !pt.contourFeatures.empty();
}

bool ShapeBasedMatcherImpl::setContour(cv::InputArray points, cv::InputArray gradients) {
    m_pending.clear();
    m_templateMeta.clear();
    m_models.clear();
    m_isTrained = false;
    m_templateImg.release();
    m_mask.release();

    PendingTemplate pt;
    pt.name = "";
    pt.fromContour = true;
    pt.originX = m_originX;
    pt.originY = m_originY;
    pt.originSet = m_originCustom;
    if (!parseContourInput(points, gradients, pt)) return false;
    if (m_contourRefW > 0 && m_contourRefH > 0) {
        pt.contourRefW = m_contourRefW;
        pt.contourRefH = m_contourRefH;
    }
    m_pending.push_back(std::move(pt));
    return true;
}

void ShapeBasedMatcherImpl::setContourRefSize(int width, int height) {
    m_contourRefW = std::max(0, width);
    m_contourRefH = std::max(0, height);
}

void ShapeBasedMatcherImpl::setCropTemplateToMask(bool enable) { m_cropTemplateToMask = enable; }
bool ShapeBasedMatcherImpl::getCropTemplateToMask() const { return m_cropTemplateToMask; }

void ShapeBasedMatcherImpl::buildScalePairs(std::vector<std::pair<float, float>>& out) const {
    out.clear();
    auto addRange = [](double mn, double mx, double step, std::vector<float>& vals) {
        vals.clear();
        if (mx <= mn + 1e-6 || step <= 0.0)
            vals.push_back((float)mn);
        else
            for (double s = mn; s <= mx + 1e-6; s += step)
                vals.push_back((float)s);
    };

    if (m_anisoScaleEnabled) {
        std::vector<float> rs, cs;
        addRange(m_minScaleR, m_maxScaleR, m_scaleStepR, rs);
        addRange(m_minScaleC, m_maxScaleC, m_scaleStepC, cs);
        for (float r : rs)
            for (float c : cs)
                out.emplace_back(r, c);
    } else {
        if (m_maxScale <= m_minScale + 1e-6 || m_scaleStep <= 0.0)
            out.emplace_back((float)m_minScale, (float)m_minScale);
        else
            for (double s = m_minScale; s <= m_maxScale + 1e-6; s += m_scaleStep)
                out.emplace_back((float)s, (float)s);
    }
}

bool ShapeBasedMatcherImpl::finalizePoseModel(ShapePoseModel& pm, BaseLevel* baseLevels,
                                                int nLevels, float ang,
                                                float scR, float scC, int templateId)
{
    pm.angle = ang;
    pm.scaleR = scR;
    pm.scaleC = scC;
    pm.scale = std::sqrt(scR * scC);
    pm.templateId = templateId;
    pm.levels.resize(nLevels);

    for (int l = 0; l < nLevels; ++l) {
        const BaseLevel& bl = baseLevels[l];
        ShapeModelLevel& ml = pm.levels[l];
        rotateFeatures(bl.features, bl.origin, ang, ml.features);

        float minX =  std::numeric_limits<float>::max();
        float minY =  std::numeric_limits<float>::max();
        float maxX = -std::numeric_limits<float>::max();
        float maxY = -std::numeric_limits<float>::max();
        for (const auto& f : ml.features) {
            minX = std::min(minX, f.dx); minY = std::min(minY, f.dy);
            maxX = std::max(maxX, f.dx); maxY = std::max(maxY, f.dy);
        }
        if (ml.features.empty()) {
            ml.width = ml.height = 0;
            ml.originX = ml.originY = 0.f;
        } else {
            const float bx = std::floor(minX);
            const float by = std::floor(minY);
            for (auto& f : ml.features) {
                f.dx -= bx;
                f.dy -= by;
            }
            ml.width  = (int)std::ceil(maxX - bx) + 1;
            ml.height = (int)std::ceil(maxY - by) + 1;
            ml.originX = bl.origin.x - bx;
            ml.originY = bl.origin.y - by;
        }
        ml.cacheFeatureOffsets();
    }
    return true;
}

void ShapeBasedMatcherImpl::icpRefinePose(
    const cv::Mat& nx, const cv::Mat& ny, const cv::Mat& mag,
    float& x, float& y, float& angleDeg,
    const ShapeModelLevel& ml,
    float magFloor, ShapeMatchMetric metric) const
{
    if (ml.features.empty() || m_icpMaxIterations <= 0) return;

    const int stride = std::max(1, (int)ml.features.size() / 64);

    for (int iter = 0; iter < m_icpMaxIterations; ++iter) {
        const float rad = angleDeg * (float)kDeg2Rad;
        const float c = std::cos(rad), s = std::sin(rad);

        double sumDx = 0, sumDy = 0, sumDa = 0;
        int count = 0;

        for (int i = 0; i < (int)ml.features.size(); i += stride) {
            const ShapeEdgeFeature& f = ml.features[i];
            const float lx = f.dx + ml.originX;
            const float ly = f.dy + ml.originY;
            const float cx = x + c * lx - s * ly;
            const float cy = y + s * lx + c * ly;

            float bestDot = -2.f;
            float bestOx = cx, bestOy = cy;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    float snx, sny, sm;
                    if (!sampleGradBilinear(nx, ny, mag, cx + dx, cy + dy, snx, sny, sm))
                        continue;
                    if (sm < magFloor) continue;
                    float d = f.gx * snx + f.gy * sny;
                    if (metric == ShapeMatchMetric::IgnoreLocalPolarity)
                        d = std::fabs(d);
                    if (d > bestDot) {
                        bestDot = d;
                        bestOx = cx + dx;
                        bestOy = cy + dy;
                    }
                }
            }
            if (bestDot < -1.f) continue;

            sumDx += bestOx - cx;
            sumDy += bestOy - cy;

            float snx, sny, sm;
            if (sampleGradBilinear(nx, ny, mag, bestOx, bestOy, snx, sny, sm) && sm >= magFloor) {
                sumDa += (double)(f.gx * sny - f.gy * snx);
            }
            ++count;
        }

        if (count == 0) break;
        const double inv = 1.0 / count;
        x += (float)(sumDx * inv);
        y += (float)(sumDy * inv);
        angleDeg += (float)(sumDa * inv * 57.2957795 * 0.2);

        if (std::abs(sumDx * inv) < 0.04 && std::abs(sumDy * inv) < 0.04 &&
            std::abs(sumDa * inv) < 0.008)
            break;
    }
}

void ShapeBasedMatcherImpl::setNumPyramidLevels(int n)          { m_numPyramidLevels = n; }
int  ShapeBasedMatcherImpl::getNumPyramidLevels() const         { return m_numPyramidLevels; }
int  ShapeBasedMatcherImpl::getEffectiveNumPyramidLevels() const { return m_actualNumLevels; }
void ShapeBasedMatcherImpl::setGreediness(double g)             { m_greediness = std::max(0.0, std::min(1.0, g)); }
double ShapeBasedMatcherImpl::getGreediness() const             { return m_greediness; }
void ShapeBasedMatcherImpl::setMaxOverlap(double o)             { m_maxOverlap = std::max(0.0, std::min(1.0, o)); }
double ShapeBasedMatcherImpl::getMaxOverlap() const             { return m_maxOverlap; }
void ShapeBasedMatcherImpl::setUseSubPixel(bool e) {
    m_subPixelMode = e ? ShapeSubPixelMode::LeastSquares : ShapeSubPixelMode::None;
}
bool ShapeBasedMatcherImpl::getUseSubPixel() const {
    return m_subPixelMode != ShapeSubPixelMode::None;
}
void ShapeBasedMatcherImpl::setSubPixelMode(ShapeSubPixelMode mode) { m_subPixelMode = mode; }
ShapeSubPixelMode ShapeBasedMatcherImpl::getSubPixelMode() const { return m_subPixelMode; }
void ShapeBasedMatcherImpl::setFindTimeoutMs(int ms) { m_findTimeoutMs = std::max(0, ms); }
int  ShapeBasedMatcherImpl::getFindTimeoutMs() const { return m_findTimeoutMs; }
void ShapeBasedMatcherImpl::setMinContrast(float c)             { m_minContrast = std::max(0.0f, c); }
float ShapeBasedMatcherImpl::getMinContrast() const             { return m_minContrast; }
void ShapeBasedMatcherImpl::setNumFeatures(int n)               { m_numFeatures = std::max(8, n); }
int  ShapeBasedMatcherImpl::getNumFeatures() const              { return m_numFeatures; }
void ShapeBasedMatcherImpl::setMetric(ShapeMatchMetric m) {
    m_metric = m;
    m_metricUserSet = true;
}
ShapeMatchMetric ShapeBasedMatcherImpl::getMetric() const       { return m_metric; }
void ShapeBasedMatcherImpl::setUsePolarity(bool e) {
    m_metric = e ? ShapeMatchMetric::UsePolarity : ShapeMatchMetric::IgnoreLocalPolarity;
    m_metricUserSet = true;
}
bool ShapeBasedMatcherImpl::getUsePolarity() const {
    return m_metric == ShapeMatchMetric::UsePolarity;
}
bool ShapeBasedMatcherImpl::isTrained() const                   { return m_isTrained; }
int  ShapeBasedMatcherImpl::numTemplates() const                { return numModels(); }

int ShapeBasedMatcherImpl::findBestPoseModel(double angle, double scaleR, double scaleC,
                                             int modelIndex) const
{
    int best = -1;
    double bestDiff = std::numeric_limits<double>::max();
    for (int i = 0; i < (int)m_models.size(); ++i) {
        if (modelIndex >= 0 && m_models[i].templateId != modelIndex) continue;
        const double dAng = std::abs((double)m_models[i].angle - angle);
        const double dR = std::abs((double)m_models[i].scaleR - scaleR);
        const double dC = std::abs((double)m_models[i].scaleC - scaleC);
        const double d = dAng + (dR + dC) * 100.0;
        if (d < bestDiff) { bestDiff = d; best = i; }
    }
    return best;
}

int ShapeBasedMatcherImpl::getFeatures(double angle, int level, cv::OutputArray out,
                                       double scaleR, double scaleC, int modelIndex) const {
    if (!m_isTrained || m_models.empty()) { out.release(); return 0; }

    const int best = findBestPoseModel(angle, scaleR, scaleC, modelIndex);
    if (best < 0) { out.release(); return 0; }

    const ShapePoseModel& pm = m_models[best];
    if (pm.levels.empty()) { out.release(); return 0; }
    int lv = std::max(0, std::min(level, (int)pm.levels.size() - 1));
    const ShapeModelLevel& ml = pm.levels[lv];

    const int n = (int)ml.features.size();
    if (n == 0) { out.release(); return 0; }

    const float pyrScale = (float)(1 << lv);
    out.create(n, 4, CV_32F);
    cv::Mat m = out.getMat();
    for (int i = 0; i < n; ++i) {
        const ShapeEdgeFeature& f = ml.features[i];
        float* r = m.ptr<float>(i);
        r[0] = (f.dx - ml.originX) * pyrScale;
        r[1] = (f.dy - ml.originY) * pyrScale;
        r[2] = f.gx;
        r[3] = f.gy;
    }
    return n;
}

int ShapeBasedMatcherImpl::getShapeModelContours(
    double angle, int level, cv::OutputArray out,
    double scaleR, double scaleC, int modelIndex,
    double poseX, double poseY, double poseAngleDeg,
    bool transformToImage) const
{
    if (!m_isTrained || m_models.empty()) { out.release(); return 0; }

    const int best = findBestPoseModel(angle, scaleR, scaleC, modelIndex);
    if (best < 0) { out.release(); return 0; }

    const ShapePoseModel& pm = m_models[best];
    if (pm.levels.empty()) { out.release(); return 0; }
    const int lv = std::max(0, std::min(level, (int)pm.levels.size() - 1));
    const ShapeModelLevel& ml = pm.levels[lv];

    const int n = (int)ml.features.size();
    if (n == 0) { out.release(); return 0; }

    const float pyrScale = (float)(1 << lv);
    const float rad = (float)(poseAngleDeg * kDeg2Rad);
    const float c = std::cos(rad);
    const float s = std::sin(rad);
    const float px = (float)poseX;
    const float py = (float)poseY;

    out.create(n, 2, CV_32F);
    cv::Mat m = out.getMat();
    for (int i = 0; i < n; ++i) {
        const ShapeEdgeFeature& f = ml.features[i];
        const float lx = (f.dx - ml.originX) * pyrScale;
        const float ly = (f.dy - ml.originY) * pyrScale;
        float* r = m.ptr<float>(i);
        if (transformToImage) {
            r[0] = px + c * lx - s * ly;
            r[1] = py + s * lx + c * ly;
        } else {
            r[0] = lx;
            r[1] = ly;
        }
    }
    return n;
}

// ---- training ----

bool ShapeBasedMatcherImpl::trainOneImage(const PendingTemplate& pt, int templateId, cv::Mat gray) {
    cv::Mat mask = pt.mask;
    double ox = pt.originSet ? pt.originX : gray.cols / 2.0;
    double oy = pt.originSet ? pt.originY : gray.rows / 2.0;
    if (!mask.empty() && m_cropTemplateToMask &&
        !cropGrayMaskToRoi(gray, mask, ox, oy, pt.originSet))
        return false;

    const double originX = ox;
    const double originY = oy;

    int requestedLevels = m_numPyramidLevels;
    m_actualNumLevels = resolveActualNumLevels(
        requestedLevels, gray.cols, gray.rows, !m_forceStrictPyramidLevels);

    if (templateId == 0) {
        m_effectiveAngleStep = (m_angleStep <= kAutoAngleStep + 1e-12)
            ? computeAutoAngleStep(gray.cols, gray.rows)
            : m_angleStep;
    }

    std::vector<std::pair<float, float>> scalePairs;
    buildScalePairs(scalePairs);

    std::vector<float> angles;
    if (m_maxAngle <= m_minAngle + 1e-6) {
        angles.push_back((float)m_minAngle);
    } else {
        for (double a = m_minAngle; a <= m_maxAngle + 1e-6; a += m_effectiveAngleStep)
            angles.push_back((float)a);
    }

    ShapeTemplateMeta meta;
    meta.name = pt.name;
    meta.id = templateId;
    meta.refW = gray.cols;
    meta.refH = gray.rows;
    meta.originX = (float)ox;
    meta.originY = (float)oy;
    m_templateMeta.push_back(meta);

    if (templateId == 0) {
        m_templateRefW = gray.cols;
        m_templateRefH = gray.rows;
    }

    m_models.reserve(m_models.size() + scalePairs.size() * angles.size());

    for (const auto& scPair : scalePairs) {
        const float scR = scPair.first;
        const float scC = scPair.second;

        cv::Mat scaledMask;
        cv::Mat scaledGray = m_anisoScaleEnabled
            ? anisoScaleTemplate(gray, mask, scR, scC, scaledMask)
            : scaleTemplate(gray, mask, scR, scaledMask);

        const double diag = std::sqrt((double)scaledGray.cols * scaledGray.cols +
                                      (double)scaledGray.rows * scaledGray.rows);
        int padding = (int)std::ceil((diag - std::min(scaledGray.cols, scaledGray.rows)) / 2.0) + 8;
        if (padding < 8) padding = 8;

        const float originPadX = padding + (float)(originX * scC);
        const float originPadY = padding + (float)(originY * scR);

        cv::Mat padTempl(scaledGray.rows + 2 * padding, scaledGray.cols + 2 * padding,
                         CV_8U, cv::Scalar(0));
        scaledGray.copyTo(padTempl(cv::Rect(padding, padding, scaledGray.cols, scaledGray.rows)));

        cv::Mat padMask(scaledMask.rows + 2 * padding, scaledMask.cols + 2 * padding,
                        CV_8U, cv::Scalar(0));
        scaledMask.copyTo(padMask(cv::Rect(padding, padding, scaledMask.cols, scaledMask.rows)));

        std::vector<BaseLevel> baseLevels(m_actualNumLevels);
        cv::Mat curTempl = padTempl;
        cv::Mat curMask  = padMask;
        cv::Size origInLevel(scaledGray.cols, scaledGray.rows);

        for (int l = 0; l < m_actualNumLevels; ++l) {
            if (l > 0) {
                cv::Mat dt, dm;
                cv::pyrDown(curTempl, dt);
                cv::pyrDown(curMask,  dm);
                cv::threshold(dm, dm, 127, 255, cv::THRESH_BINARY);
                curTempl = dt; curMask = dm;
                origInLevel.width  = (origInLevel.width  + 1) / 2;
                origInLevel.height = (origInLevel.height + 1) / 2;
            }

            BaseLevel& bl = baseLevels[l];
            bl.sz = curTempl.size();
            const float inv = 1.f / (float)(1 << l);
            bl.origin = cv::Point2f(originPadX * inv, originPadY * inv);
            bl.width   = origInLevel.width;
            bl.height  = origInLevel.height;

            int nf = pyramidFeatureBudget(m_numFeatures, l);
            extractEdgeFeatures(curTempl, curMask, nf, m_minContrast,
                                bl.width, bl.height, bl.features);
        }

        for (float ang : angles) {
            ShapePoseModel pm;
            if (!finalizePoseModel(pm, baseLevels.data(), m_actualNumLevels,
                                   ang, scR, scC, templateId))
                return false;
            m_models.push_back(std::move(pm));
        }
    }
    return true;
}

bool ShapeBasedMatcherImpl::trainOneContour(const PendingTemplate& pt, int templateId) {
    const double ox = pt.originSet ? pt.originX : pt.contourRefW / 2.0;
    const double oy = pt.originSet ? pt.originY : pt.contourRefH / 2.0;

    int requestedLevels = m_numPyramidLevels;
    m_actualNumLevels = resolveActualNumLevels(
        requestedLevels, pt.contourRefW, pt.contourRefH, !m_forceStrictPyramidLevels);

    if (templateId == 0) {
        m_effectiveAngleStep = (m_angleStep <= kAutoAngleStep + 1e-12)
            ? computeAutoAngleStep(pt.contourRefW, pt.contourRefH)
            : m_angleStep;
    }

    std::vector<std::pair<float, float>> scalePairs;
    buildScalePairs(scalePairs);

    std::vector<float> angles;
    if (m_maxAngle <= m_minAngle + 1e-6) {
        angles.push_back((float)m_minAngle);
    } else {
        for (double a = m_minAngle; a <= m_maxAngle + 1e-6; a += m_effectiveAngleStep)
            angles.push_back((float)a);
    }

    ShapeTemplateMeta meta;
    meta.name = pt.name;
    meta.id = templateId;
    meta.refW = pt.contourRefW;
    meta.refH = pt.contourRefH;
    meta.originX = (float)ox;
    meta.originY = (float)oy;
    m_templateMeta.push_back(meta);

    if (templateId == 0) {
        m_templateRefW = pt.contourRefW;
        m_templateRefH = pt.contourRefH;
    }

    m_models.reserve(m_models.size() + scalePairs.size() * angles.size());

    for (const auto& scPair : scalePairs) {
        const float scR = scPair.first;
        const float scC = scPair.second;

        const int refW = std::max(1, (int)std::lround(pt.contourRefW * scC));
        const int refH = std::max(1, (int)std::lround(pt.contourRefH * scR));
        const double diag = std::sqrt((double)refW * refW + (double)refH * refH);
        int padding = (int)std::ceil((diag - std::min(refW, refH)) / 2.0) + 8;
        if (padding < 8) padding = 8;

        std::vector<ShapeEdgeFeature> scaledFeats;
        scaledFeats.reserve(pt.contourFeatures.size());
        for (const auto& f : pt.contourFeatures) {
            ShapeEdgeFeature sf = f;
            // Contour points are model-relative (Halcon); convert to template pixel coords.
            sf.dx = (f.dx + (float)ox) * scC + (float)padding;
            sf.dy = (f.dy + (float)oy) * scR + (float)padding;
            scaledFeats.push_back(sf);
        }

        const float originPadX = padding + (float)(ox * scC);
        const float originPadY = padding + (float)(oy * scR);

        std::vector<BaseLevel> baseLevels(m_actualNumLevels);
        cv::Size origInLevel(refW, refH);
        for (int l = 0; l < m_actualNumLevels; ++l) {
            const float inv = 1.f / (float)(1 << l);
            BaseLevel& bl = baseLevels[l];
            bl.origin = cv::Point2f(originPadX * inv, originPadY * inv);
            bl.features.clear();
            bl.features.reserve(scaledFeats.size());
            for (const auto& f : scaledFeats) {
                ShapeEdgeFeature lf = f;
                lf.dx *= inv;
                lf.dy *= inv;
                bl.features.push_back(lf);
            }
            bl.width  = std::max(1, (origInLevel.width  + (1 << l) - 1) >> l);
            bl.height = std::max(1, (origInLevel.height + (1 << l) - 1) >> l);
            bl.sz = cv::Size(bl.width, bl.height);
        }

        for (float ang : angles) {
            ShapePoseModel pm;
            if (!finalizePoseModel(pm, baseLevels.data(), m_actualNumLevels,
                                   ang, scR, scC, templateId))
                return false;
            m_models.push_back(std::move(pm));
        }
    }
    return true;
}

bool ShapeBasedMatcherImpl::trainOne(const PendingTemplate& pt, int templateId) {
    if (pt.fromContour)
        return trainOneContour(pt, templateId);
    cv::Mat gray = ensureGray8u(pt.image);
    if (gray.empty()) return false;
    return trainOneImage(pt, templateId, gray);
}

bool ShapeBasedMatcherImpl::validatePyramidDepthSelfMatch(const cv::Mat& trainGray)
{
    if (trainGray.empty() || m_models.empty()) return true;

    m_isTrained = true;
    const double savedG = m_greediness;
    m_greediness = 0.0;
    const std::vector<MatchInfo> hits = find(trainGray, 0.35f, 1);
    m_greediness = savedG;
    return !hits.empty() && hits[0].score >= 0.35f;
}

bool ShapeBasedMatcherImpl::train() {
    if (m_pending.empty()) return false;

    clearSceneGradientCache();
    m_models.clear();
    m_templateMeta.clear();
    m_isTrained = false;
    m_forceStrictPyramidLevels = false;

    const int requestedLevels = m_numPyramidLevels;

    for (int i = 0; i < (int)m_pending.size(); ++i) {
        if (!trainOne(m_pending[i], i)) return false;
    }

    // Self-match gate: retry strict; drop one pyramid level if still failing.
    if (m_pending.size() == 1 && requestedLevels >= 2) {
        cv::Mat trainGray = ensureGray8u(m_pending[0].image);
        if (!trainGray.empty() && !validatePyramidDepthSelfMatch(trainGray)) {
            clearSceneGradientCache();
            m_models.clear();
            m_templateMeta.clear();
            m_isTrained = false;
            m_forceStrictPyramidLevels = true;
            if (!trainOne(m_pending[0], 0)) {
                m_forceStrictPyramidLevels = false;
                return false;
            }
            m_forceStrictPyramidLevels = false;
            if (!validatePyramidDepthSelfMatch(trainGray)
                && requestedLevels > 2) {
                clearSceneGradientCache();
                m_models.clear();
                m_templateMeta.clear();
                m_isTrained = false;
                m_forceStrictPyramidLevels = true;
                m_numPyramidLevels = requestedLevels - 1;
                if (!trainOne(m_pending[0], 0)) {
                    m_forceStrictPyramidLevels = false;
                    return false;
                }
                m_forceStrictPyramidLevels = false;
            }
        }
    }

    if (m_pending.size() == 1) {
        m_templateImg = m_pending[0].image.clone();
        m_mask = m_pending[0].mask.clone();
    }

    // P34 Hofhauser 2009 / Solution Guide II-B §3.2.6.2: elongated belt/metal —
    // ignore_local_polarity unless caller set metric explicitly.
    if (!m_metricUserSet && !m_templateMeta.empty()) {
        const int rw = std::max(1, m_templateMeta[0].refW);
        const int rh = std::max(1, m_templateMeta[0].refH);
        if (rw >= rh * 3 || rh >= rw * 3)
            m_metric = ShapeMatchMetric::IgnoreLocalPolarity;
    }

    m_isTrained = !m_models.empty();
    return m_isTrained;
}

int ShapeBasedMatcherImpl::inspectModel(int modelIndex,
                                        std::vector<cv::Mat>& levelImages,
                                        std::vector<int>& featureCounts) const
{
    levelImages.clear();
    featureCounts.clear();
    if (!m_isTrained || modelIndex < 0 || modelIndex >= (int)m_templateMeta.size())
        return 0;

    // Find a pose model for this template (prefer angle=0, scale=1).
    int poseIdx = -1;
    for (int i = 0; i < (int)m_models.size(); ++i) {
        if (m_models[i].templateId != modelIndex) continue;
        if (std::abs(m_models[i].angle) < 1e-3f && std::abs(m_models[i].scale - 1.f) < 1e-3f) {
            poseIdx = i; break;
        }
        if (poseIdx < 0) poseIdx = i;
    }
    if (poseIdx < 0) return 0;

    const ShapePoseModel& pm = m_models[poseIdx];
    const int nLevels = (int)pm.levels.size();
    if (nLevels <= 0) return 0;

    cv::Mat gray;
    if (modelIndex == 0 && !m_templateImg.empty())
        gray = ensureGray8u(m_templateImg);
    else if (modelIndex < (int)m_pending.size())
        gray = ensureGray8u(m_pending[modelIndex].image);

    cv::Mat curGray;
    if (!gray.empty()) curGray = gray;

    for (int l = 0; l < nLevels; ++l) {
        const ShapeModelLevel& ml = pm.levels[l];
        featureCounts.push_back((int)ml.features.size());

        cv::Mat vis;
        if (!curGray.empty()) {
            if (l > 0) {
                cv::Mat down;
                cv::pyrDown(curGray, down);
                curGray = down;
            }
            cv::cvtColor(curGray, vis, cv::COLOR_GRAY2BGR);
        } else {
            vis = cv::Mat(ml.height, ml.width, CV_8UC3, cv::Scalar(32, 32, 32));
        }

        const float pyrScale = (float)(1 << l);
        for (const auto& f : ml.features) {
            int px = (int)std::lround(f.dx * pyrScale + ml.originX * pyrScale);
            int py = (int)std::lround(f.dy * pyrScale + ml.originY * pyrScale);
            if (px >= 0 && px < vis.cols && py >= 0 && py < vis.rows)
                cv::circle(vis, cv::Point(px, py), 1, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
        }
        levelImages.push_back(std::move(vis));
    }

    if (!levelImages.empty()) {
        int maxW = 0;
        for (const auto& im : levelImages)
            maxW = std::max(maxW, im.cols);
        for (auto& im : levelImages) {
            if (im.cols < maxW) {
                cv::Mat padded;
                cv::copyMakeBorder(im, padded, 0, 0, 0, maxW - im.cols,
                                   cv::BORDER_CONSTANT, cv::Scalar(16, 16, 16));
                im = padded;
            }
        }
    }
    return (int)levelImages.size();
}

// ---- save / load -------------------------------------------------------------

bool ShapeBasedMatcherImpl::saveModel(const std::string& path) const {
    if (!m_isTrained) return false;
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;

    fs << "format" << "edge_shape_v5";
    fs << "min_angle" << m_minAngle;
    fs << "max_angle" << m_maxAngle;
    fs << "angle_step" << m_angleStep;
    fs << "effective_angle_step" << m_effectiveAngleStep;
    fs << "min_scale" << m_minScale;
    fs << "max_scale" << m_maxScale;
    fs << "scale_step" << m_scaleStep;
    fs << "aniso_scale_enabled" << (m_anisoScaleEnabled ? 1 : 0);
    fs << "min_scale_r" << m_minScaleR;
    fs << "max_scale_r" << m_maxScaleR;
    fs << "scale_step_r" << m_scaleStepR;
    fs << "min_scale_c" << m_minScaleC;
    fs << "max_scale_c" << m_maxScaleC;
    fs << "scale_step_c" << m_scaleStepC;
    fs << "num_pyramid_levels" << m_numPyramidLevels;
    fs << "actual_num_levels" << m_actualNumLevels;
    fs << "min_contrast" << m_minContrast;
    fs << "num_features" << m_numFeatures;
    fs << "metric" << (int)m_metric;
    fs << "sub_pixel_mode" << (int)m_subPixelMode;
    fs << "find_timeout_ms" << m_findTimeoutMs;
    fs << "use_icp_refine" << (m_useIcpRefine ? 1 : 0);
    fs << "icp_max_iterations" << m_icpMaxIterations;
    fs << "template_ref_w" << m_templateRefW;
    fs << "template_ref_h" << m_templateRefH;
    fs << "position_bias_x" << m_positionBiasX;
    fs << "position_bias_y" << m_positionBiasY;

    fs << "templates" << "[";
    for (const auto& tm : m_templateMeta) {
        fs << "{";
        fs << "name" << tm.name;
        fs << "id" << tm.id;
        fs << "ref_w" << tm.refW << "ref_h" << tm.refH;
        fs << "origin_x" << tm.originX << "origin_y" << tm.originY;
        fs << "}";
    }
    fs << "]";

    fs << "models" << "[";
    for (const auto& pm : m_models) {
        fs << "{";
        fs << "angle" << pm.angle;
        fs << "scale" << pm.scale;
        fs << "scale_r" << pm.scaleR;
        fs << "scale_c" << pm.scaleC;
        fs << "template_id" << pm.templateId;
        fs << "levels" << "[";
        for (const auto& ml : pm.levels) {
            fs << "{";
            fs << "w" << ml.width << "h" << ml.height;
            fs << "ox" << ml.originX << "oy" << ml.originY;
            // Pack features as a flat float matrix [N x 4]: dx, dy, gx, gy.
            const int n = (int)ml.features.size();
            cv::Mat packed(n, 4, CV_32F);
            for (int i = 0; i < n; ++i) {
                float* r = packed.ptr<float>(i);
                r[0] = ml.features[i].dx;
                r[1] = ml.features[i].dy;
                r[2] = ml.features[i].gx;
                r[3] = ml.features[i].gy;
            }
            fs << "feat" << packed;
            fs << "}";
        }
        fs << "]";
        fs << "}";
    }
    fs << "]";
    return true;
}

bool ShapeBasedMatcherImpl::loadModel(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;

    std::string fmt;
    fs["format"] >> fmt;
    // v1 is forward-compatible: features were already serialized as floats and the
    // new use_polarity flag defaults to true (polarity-sensitive) when absent.
    if (fmt != "edge_shape_v5" && fmt != "edge_shape_v4" && fmt != "edge_shape_v3" &&
        fmt != "edge_shape_v2" && fmt != "edge_shape_v1") return false;

    fs["min_angle"]          >> m_minAngle;
    fs["max_angle"]          >> m_maxAngle;
    fs["angle_step"]         >> m_angleStep;
    if (!fs["effective_angle_step"].empty())
        fs["effective_angle_step"] >> m_effectiveAngleStep;
    else
        m_effectiveAngleStep = m_angleStep > 0 ? m_angleStep : 1.0;

    if (!fs["min_scale"].empty()) fs["min_scale"] >> m_minScale;
    else m_minScale = 1.0;
    if (!fs["max_scale"].empty()) fs["max_scale"] >> m_maxScale;
    else m_maxScale = m_minScale;
    if (!fs["scale_step"].empty()) fs["scale_step"] >> m_scaleStep;
    else m_scaleStep = 0.0;

    if (!fs["aniso_scale_enabled"].empty()) {
        int aniso = 0;
        fs["aniso_scale_enabled"] >> aniso;
        m_anisoScaleEnabled = aniso != 0;
    } else {
        m_anisoScaleEnabled = false;
    }
    if (!fs["min_scale_r"].empty()) fs["min_scale_r"] >> m_minScaleR;
    else m_minScaleR = m_minScale;
    if (!fs["max_scale_r"].empty()) fs["max_scale_r"] >> m_maxScaleR;
    else m_maxScaleR = m_maxScale;
    if (!fs["scale_step_r"].empty()) fs["scale_step_r"] >> m_scaleStepR;
    else m_scaleStepR = m_scaleStep;
    if (!fs["min_scale_c"].empty()) fs["min_scale_c"] >> m_minScaleC;
    else m_minScaleC = m_minScale;
    if (!fs["max_scale_c"].empty()) fs["max_scale_c"] >> m_maxScaleC;
    else m_maxScaleC = m_maxScale;
    if (!fs["scale_step_c"].empty()) fs["scale_step_c"] >> m_scaleStepC;
    else m_scaleStepC = m_scaleStep;

    fs["num_pyramid_levels"] >> m_numPyramidLevels;
    fs["actual_num_levels"]  >> m_actualNumLevels;
    m_positionBiasX = 0.0;
    m_positionBiasY = 0.0;
    if (!fs["position_bias_x"].empty()) fs["position_bias_x"] >> m_positionBiasX;
    if (!fs["position_bias_y"].empty()) fs["position_bias_y"] >> m_positionBiasY;
    fs["min_contrast"]       >> m_minContrast;
    fs["num_features"]       >> m_numFeatures;

    if (!fs["metric"].empty()) {
        int met = 0;
        fs["metric"] >> met;
        m_metric = static_cast<ShapeMatchMetric>(met);
        m_metricUserSet = true;
    } else {
        int pol = 1;
        if (!fs["use_polarity"].empty()) fs["use_polarity"] >> pol;
        m_metric = pol ? ShapeMatchMetric::UsePolarity : ShapeMatchMetric::IgnoreLocalPolarity;
        if (!fs["use_polarity"].empty())
            m_metricUserSet = true;
    }

    if (!fs["sub_pixel_mode"].empty()) {
        int spm = (int)ShapeSubPixelMode::LeastSquares;
        fs["sub_pixel_mode"] >> spm;
        m_subPixelMode = static_cast<ShapeSubPixelMode>(spm);
    } else {
        int sp = 1;
        if (!fs["use_sub_pixel"].empty()) fs["use_sub_pixel"] >> sp;
        m_subPixelMode = sp ? ShapeSubPixelMode::LeastSquares : ShapeSubPixelMode::None;
    }

    if (!fs["find_timeout_ms"].empty())
        fs["find_timeout_ms"] >> m_findTimeoutMs;
    else
        m_findTimeoutMs = 0;

    if (!fs["use_icp_refine"].empty()) {
        int icp = 0;
        fs["use_icp_refine"] >> icp;
        m_useIcpRefine = icp != 0;
    } else {
        m_useIcpRefine = false;
    }
    if (!fs["icp_max_iterations"].empty())
        fs["icp_max_iterations"] >> m_icpMaxIterations;
    else
        m_icpMaxIterations = 5;

    fs["template_ref_w"]     >> m_templateRefW;
    fs["template_ref_h"]     >> m_templateRefH;

    m_templateMeta.clear();
    cv::FileNode templates = fs["templates"];
    if (!templates.empty()) {
        for (auto tit = templates.begin(); tit != templates.end(); ++tit) {
            ShapeTemplateMeta tm;
            (*tit)["name"] >> tm.name;
            if (!(*tit)["id"].empty()) (*tit)["id"] >> tm.id;
            else tm.id = (int)m_templateMeta.size();
            (*tit)["ref_w"] >> tm.refW;
            (*tit)["ref_h"] >> tm.refH;
            if (!(*tit)["origin_x"].empty()) (*tit)["origin_x"] >> tm.originX;
            else tm.originX = tm.refW / 2.f;
            if (!(*tit)["origin_y"].empty()) (*tit)["origin_y"] >> tm.originY;
            else tm.originY = tm.refH / 2.f;
            m_templateMeta.push_back(std::move(tm));
        }
    } else {
        ShapeTemplateMeta tm;
        tm.name = "";
        tm.id = 0;
        tm.refW = m_templateRefW;
        tm.refH = m_templateRefH;
        tm.originX = m_templateRefW / 2.f;
        tm.originY = m_templateRefH / 2.f;
        m_templateMeta.push_back(tm);
    }

    m_models.clear();
    cv::FileNode models = fs["models"];
    for (auto mit = models.begin(); mit != models.end(); ++mit) {
        ShapePoseModel pm;
        (*mit)["angle"] >> pm.angle;
        if (!(*mit)["scale"].empty())
            (*mit)["scale"] >> pm.scale;
        else
            pm.scale = 1.0f;
        if (!(*mit)["scale_r"].empty())
            (*mit)["scale_r"] >> pm.scaleR;
        else
            pm.scaleR = pm.scale;
        if (!(*mit)["scale_c"].empty())
            (*mit)["scale_c"] >> pm.scaleC;
        else
            pm.scaleC = pm.scale;
        if (!(*mit)["template_id"].empty())
            (*mit)["template_id"] >> pm.templateId;
        else
            pm.templateId = 0;
        cv::FileNode levels = (*mit)["levels"];
        for (auto lit = levels.begin(); lit != levels.end(); ++lit) {
            ShapeModelLevel ml;
            (*lit)["w"]  >> ml.width;
            (*lit)["h"]  >> ml.height;
            (*lit)["ox"] >> ml.originX;
            (*lit)["oy"] >> ml.originY;
            cv::Mat packed;
            (*lit)["feat"] >> packed;
            ml.features.resize(packed.rows);
            for (int i = 0; i < packed.rows; ++i) {
                const float* r = packed.ptr<float>(i);
                ml.features[i].dx = r[0];
                ml.features[i].dy = r[1];
                ml.features[i].gx = r[2];
                ml.features[i].gy = r[3];
            }
            ml.cacheFeatureOffsets();
            pm.levels.push_back(std::move(ml));
        }
        m_models.push_back(std::move(pm));
    }

    m_isTrained = !m_models.empty();
    return m_isTrained;
}

// ---- matching ---------------------------------------------------------------

std::vector<MatchInfo> ShapeBasedMatcherImpl::find(
    cv::InputArray source,
    float scoreThreshold,
    int maxTargets,
    cv::Rect searchRegion,
    cv::InputArray searchMask)
{
    if (!opencvsharp_license_runtime_activated()) return {};

    std::vector<MatchInfo> emptyOut;
    if (!m_isTrained || m_models.empty()) return emptyOut;
    if (maxTargets <= 0) return emptyOut;

    cv::Mat src = source.getMat();
    if (src.empty()) return emptyOut;
    cv::Mat srcGray = ensureGray8u(src);

    cv::Mat fullSearchMask;
    if (!searchMask.empty()) {
        cv::Mat sm = searchMask.getMat();
        if (sm.size() == srcGray.size() && sm.type() == CV_8UC1)
            fullSearchMask = sm;
    }

    // Crop to search region (P37: view when possible — parent srcGray stays in scope).
    cv::Point searchOffset(0, 0);
    if (searchRegion.width > 0 && searchRegion.height > 0) {
        cv::Rect imgRect(0, 0, srcGray.cols, srcGray.rows);
        cv::Rect clipped = searchRegion & imgRect;
        if (clipped.width <= 0 || clipped.height <= 0) return emptyOut;
        srcGray = srcGray(clipped);
        if (!fullSearchMask.empty())
            fullSearchMask = fullSearchMask(clipped);
        searchOffset = clipped.tl();
    }
    // Pyramid depth must match the trained models.
    const int nLevels = m_actualNumLevels;
    if (nLevels <= 0) return emptyOut;

    const int numModels = (int)m_models.size();

    const float greediness = (float)std::max(0.0, std::min(1.0, m_greediness));
    const int fullScenePixels = srcGray.cols * srcGray.rows;
    const bool largeScene = fullScenePixels > 480 * 480;
    const bool smallScene = fullScenePixels <= 512 * 512;
    int tplRefW = std::max(1, m_templateRefW);
    int tplRefH = std::max(1, m_templateRefH);
    if (!m_templateMeta.empty()) {
        tplRefW = std::max(1, m_templateMeta[0].refW);
        tplRefH = std::max(1, m_templateMeta[0].refH);
    }
    // P17/P18 Halcon §2.3 + §3.1: wide belt + pyramid=2 → L0-only find (skip L1 grad entirely).
    const bool beltPyr2Fast = tplRefW >= tplRefH * 3 && largeScene && nLevels >= 2
        && maxTargets == 1 && greediness >= 0.9f;

    // When the scene is template-sized (self-match / small ROI), the default top level can
    // be too coarse — model bbox equals scene extent and coarse scan would skip all poses.
    int topLevel = nLevels - 1;
    SourcePyramid sp;
    if (beltPyr2Fast) {
        topLevel = 0;
        bool anyFit = false;
        for (int mi = 0; mi < numModels; ++mi) {
            const ShapeModelLevel& ml = m_models[mi].levels[0];
            if (ml.features.empty()) continue;
            if (ml.width <= srcGray.cols && ml.height <= srcGray.rows) {
                anyFit = true;
                break;
            }
        }
        if (!anyFit) return emptyOut;
        resizeSourcePyramid(sp, nLevels);
        sp.gray[0] = srcGray;
    } else {
        for (; topLevel >= 0; --topLevel) {
            ensureScenePyramid(srcGray, nLevels, topLevel, sp);
            const cv::Mat& lvNx = sp.nx[topLevel];
            if (lvNx.empty()) continue;
            bool anyFit = false;
            for (int mi = 0; mi < numModels; ++mi) {
                const ShapeModelLevel& ml = m_models[mi].levels[topLevel];
                if (ml.features.empty()) continue;
                if (ml.width <= lvNx.cols && ml.height <= lvNx.rows) {
                    anyFit = true;
                    break;
                }
            }
            if (anyFit) break;
        }
        if (topLevel < 0) return emptyOut;
        ensureScenePyramid(srcGray, nLevels, topLevel, sp);
    }
    const int coarseStartLevel = beltPyr2Fast ? (nLevels - 1) : topLevel;
    int fineTopLevel = topLevel;
    bool beltCoarseAtL0 = false;

    std::vector<cv::Mat> maskPyramid;
    buildMaskPyramid(fullSearchMask, nLevels, maskPyramid);
    const bool useSearchMask = !maskPyramid.empty();

    auto maskAllowed = [&](int level, int x, int y, const ShapeModelLevel& ml) -> bool {
        if (!useSearchMask || level >= (int)maskPyramid.size()) return true;
        const cv::Mat& m = maskPyramid[level];
        const int ox = (int)std::lround(x + ml.originX);
        const int oy = (int)std::lround(y + ml.originY);
        if (ox < 0 || oy < 0 || ox >= m.cols || oy >= m.rows) return false;
        return m.at<uchar>(oy, ox) > 0;
    };

    const ShapeMatchMetric metric = m_metric;
    const float userScore01 = std::max(0.0f, std::min(1.0f, scoreThreshold));
    // P43 belt track: only trust motion hint after a reliable (>= MinScore) prior hit.
    const bool hintReliable = !m_hasFindHint || m_findHintScore + 1e-3f >= userScore01;
    // PatMax Accept (coarse) below user MinScore; Halcon Greediness interacts with both.
    const float coarseSearchScore = patmaxCoarseAcceptScore(userScore01, greediness, nLevels);

    // Halcon MinContrast at L0; coarse levels use scaled floor (pyramid blur).
    const float magFloor = m_minContrast * 0.5f;
    const float fineMagFloor = m_minContrast;

    const float coarseMagFloorTop = coarseMagFloor(magFloor, nLevels, topLevel);
    const cv::Mat& topNx  = sp.nx[topLevel];
    const cv::Mat& topNy  = sp.ny[topLevel];
    const cv::Mat& topMag = sp.mag[topLevel];

    // Detect a full 360-degree sweep so angle-neighbour refinement can wrap.
    const bool angleWrap =
        numModels > 2 &&
        (m_maxAngle - m_minAngle) >= 360.0 - m_effectiveAngleStep * 0.5;

    FindTimer timer(m_findTimeoutMs);

    // Pre-cache feature offsets once before any OpenMP region. cacheFeatureOffsets reorders
    // feature vectors (invalidates raw pointers) and must not run concurrently (P20 AV fix).
    for (int mi = 0; mi < numModels; ++mi) {
        for (int l = 0; l < nLevels && l < (int)m_models[mi].levels.size(); ++l) {
            ShapeModelLevel& ml = m_models[mi].levels[l];
            if (!ml.features.empty())
                ml.cacheFeatureOffsets();
        }
    }

    // ===== STAGE 1: coarse scan at top pyramid level (PatMax/Halcon coarse filter). =====
    //
    // Integer-pixel coarse with optional stride on large scenes + local refine (Cognex
    // OptimizeFindPatMaxPatterns). Coarse Accept < MinScore; greediness early-out.
    // Retry with lowered Accept only when no candidates found.

    struct RawCand {
        int   modelIdx;
        int   x, y;
        float score;
    };
    std::vector<RawCand> topCands;

    const int stepNx = topNx.empty() ? 0 : (int)(topNx.step / sizeof(float));
    const int stepMg = topMag.empty() ? 0 : (int)(topMag.step / sizeof(float));

    const float coarseThr = coarseAcceptThreshold(coarseSearchScore, greediness, nLevels);

    const int coarseScenePixels = topNx.empty() ? fullScenePixels : topNx.cols * topNx.rows;
    int modelWTop = 0, modelHTop = 0;
    if (!m_models.empty() && topLevel >= 0 && topLevel < (int)m_models[0].levels.size()) {
        modelWTop = m_models[0].levels[topLevel].width;
        modelHTop = m_models[0].levels[topLevel].height;
    }
    int tplW = tplRefW;
    int tplH = tplRefH;
    if (topLevel > 0) {
        tplW = std::max(1, (tplW + (1 << topLevel) - 1) >> topLevel);
        tplH = std::max(1, (tplH + (1 << topLevel) - 1) >> topLevel);
    }
    const int beltWTop = std::max(modelWTop, tplW);
    const int beltHTop = std::max(modelHTop, tplH);
    const float tplAspect = (float)std::max(tplW, tplH) / (float)std::max(1, std::min(tplW, tplH));
    const bool elongHorizTpl = tplW >= tplH * 3;
    const bool elongVertTpl  = tplH >= tplW * 3;
    // Use full-resolution dims — pyramid-scaled tplW/H breaks 083933 (761px @ L0, ~190px @ L2).
    const bool largeTemplate = std::min(tplRefW, tplRefH) > 500;
    const bool mediumTemplate = !largeTemplate && std::min(tplRefW, tplRefH) > 200;
    const bool compactTemplate = !elongHorizTpl && std::max(tplRefW, tplRefH) <= 320;
    // P42: large/medium fine accept below MinScore (083933 ~0.49, 165340 ~0.49–0.50 vs Halcon hit).
    const float acceptFine01 = largeTemplate
        ? std::max(0.06f, userScore01 - 0.03f)
        : (mediumTemplate ? std::max(0.06f, userScore01 - 0.02f)
            : (compactTemplate ? std::max(0.06f, userScore01 - 0.03f) : userScore01));
    const bool elongHorizFeat = modelWTop >= modelHTop * 3;
    const bool elongVertFeat  = modelHTop >= modelWTop * 3;
    const bool elongatedTpl = tplAspect >= 3.f;

    const int finePoolCap = halconFinePoolCap(
        maxTargets, greediness, elongHorizTpl && largeScene, nLevels, smallScene);
    const float fineScanMinAvg = std::max(0.06f, userScore01 * (0.42f - 0.12f * greediness));

    // P42/P43 belt track: single coarse pass when motion hint is reliable.
    // P44a: fast track only on strong prior hit — coarse stride misses trigger P43 expand (+100ms).
    const bool beltTrackFast = elongHorizTpl && m_trackLastFind && m_hasFindHint
        && largeScene && hintReliable
        && m_findHintScore + 1e-3f >= userScore01 + 0.04f;
    const int maxCoarsePasses = beltTrackFast ? 1
        : ((maxTargets == 1 && greediness >= 0.9f && !elongHorizTpl)
            ? 1
            : ((nLevels <= 1)
                ? (smallScene ? 1 : (elongHorizTpl && largeScene ? 2 : 3))
                : 2));
    const float nearTopResurrect = (elongHorizTpl && largeScene)
        ? 0.08f
        : ((maxTargets == 1 && greediness >= 0.9f) ? 0.04f : 0.08f);
    float coarseThrUsed = coarseThr;
    int coarseScanStepUsed = 1;
    int coarseScanStepXUsed = 1;
    int coarseScanStepYUsed = 1;

    // P12 Halcon §2.4.1: belt coarse Y-band carries into fine L0 shared Sobel strip.
    int beltMotionYLo = 0;
    int beltMotionYHi = 0;
    bool haveBeltMotionY = false;
    float beltMedianPlX = -1.f;

    std::vector<CoarseModelCtx> ctxs;
    ctxs.reserve(numModels);
    int globalYMax = 0;
    if (!beltPyr2Fast) {
    for (int mi = 0; mi < numModels; ++mi) {
        const ShapeModelLevel& ml = m_models[mi].levels[topLevel];
        if (ml.features.empty() ||
            ml.width > topNx.cols ||
            ml.height > topNx.rows) continue;
        CoarseModelCtx c;
        c.mi = mi;
        c.ml = &ml;
        c.feats = ml.features.data();
        c.ox = ml.featOx.data();
        c.oy = ml.featOy.data();
        c.n = (int)ml.features.size();
        c.yMax = topNx.rows - ml.height;
        c.xMax = topNx.cols - ml.width;
        c.thr = coarseThrUsed;
        globalYMax = std::max(globalYMax, c.yMax);
        ctxs.push_back(c);
    }
    // P13: evaluate likely rotation first (Halcon fused pose order — greediness prunes faster).
    if (elongHorizTpl && largeScene && nLevels <= 1 && ctxs.size() > 1) {
        std::stable_sort(ctxs.begin(), ctxs.end(), [this](const CoarseModelCtx& a, const CoarseModelCtx& b) {
            const float aa = std::fabs(m_models[a.mi].angle);
            const float ab = std::fabs(m_models[b.mi].angle);
            return aa < ab;
        });
    }

    // Halcon §2.4.1 / PatMax: template-sized ROI — 9-anchor coarse before full grid.
    const int tplRefArea = std::max(1, m_templateRefW * m_templateRefH);
    const bool templateSizedRoi = smallScene && maxTargets == 1 && greediness >= 0.85f
        && fullScenePixels <= tplRefArea * 8 && !elongHorizTpl;
    bool anchorCoarseOk = false;
    if (templateSizedRoi && !ctxs.empty()) {
        const int xMaxA = ctxs[0].xMax;
        const int yMaxA = globalYMax;
        if (xMaxA <= 72 && yMaxA <= 72) {
            const float* nxBaseA = topNx.ptr<float>(0);
            const float* nyBaseA = topNy.ptr<float>(0);
            const float* mgBaseA = topMag.ptr<float>(0);
            const float passG = greediness;
            auto tryAnchor = [&](int x, int y) {
                int bestMi = -1;
                float bestS = -1.f;
                for (const CoarseModelCtx& c : ctxs) {
                    if (y > c.yMax || x > c.xMax) continue;
                    if (!maskAllowed(topLevel, x, y, *c.ml)) continue;
                    float accPos = 0.f, accNeg = 0.f;
                    if (!coarseScoreAccumulate(
                            x, y, c.n, c.feats, c.ox, c.oy,
                            nxBaseA, nyBaseA, mgBaseA,
                            stepNx, stepMg, coarseMagFloorTop, metric,
                            c.thr, passG, accPos, accNeg))
                        continue;
                    float s = metricScore(accPos, accNeg, c.n, metric);
                    if (s >= c.thr && s > bestS) { bestS = s; bestMi = c.mi; }
                }
                if (bestMi >= 0) {
                    RawCand rc;
                    rc.modelIdx = bestMi; rc.x = x; rc.y = y; rc.score = bestS;
                    topCands.push_back(rc);
                }
            };
            const int ax0 = 0, ax1 = xMaxA / 2, ax2 = xMaxA;
            const int ay0 = 0, ay1 = yMaxA / 2, ay2 = yMaxA;
            tryAnchor(ax0, ay0); tryAnchor(ax1, ay0); tryAnchor(ax2, ay0);
            tryAnchor(ax0, ay1); tryAnchor(ax1, ay1); tryAnchor(ax2, ay1);
            tryAnchor(ax0, ay2); tryAnchor(ax1, ay2); tryAnchor(ax2, ay2);
            if (!topCands.empty()) {
                std::sort(topCands.begin(), topCands.end(),
                    [](const RawCand& a, const RawCand& b) { return a.score > b.score; });
                if (topCands[0].score >= coarseSearchScore * 0.72f) {
                    anchorCoarseOk = true;
                    coarseScanStepUsed = coarseScanStepXUsed = coarseScanStepYUsed = 4;
                } else
                    topCands.clear();
            }
        }
    }

    if (!ctxs.empty() && !anchorCoarseOk && !beltPyr2Fast) {
        const float* nxBase = topNx.ptr<float>(0);
        const float* nyBase = topNy.ptr<float>(0);
        const float* mgBase = topMag.ptr<float>(0);

#ifdef _OPENMP
        const int coarseThreads = omp_get_max_threads();
#else
        const int coarseThreads = 1;
#endif
        std::vector<std::vector<RawCand>> coarseBuckets((size_t)coarseThreads);
        std::vector<int> beltPassYSeeds;
        beltPassYSeeds.reserve(32);

        for (int coarsePass = 0; coarsePass < maxCoarsePasses; ++coarsePass) {
            if (timer.check()) break;
            const float passGreediness =
                (coarsePass == 0 && topLevel > 0 && nLevels >= 3) ? 0.f : greediness;
            if (coarsePass > 0) {
                coarseThrUsed = std::max(0.04f, coarseSearchScore * (0.12f / (float)coarsePass));
                for (CoarseModelCtx& c : ctxs)
                    c.thr = coarseThrUsed;
            }

            int scanStepX = 1;
            int scanStepY = 1;
            // P35 Halcon §2.4.2 / PatMax coarse granularity: subsample on pyramid levels
            // (coarseScenePixels is at topLevel, not full-res — 640×480 was too high for ROI/L1).
            if (topLevel > 0 && coarsePass == 0 && coarseScenePixels > 128 * 128) {
                if (elongHorizTpl && largeScene && maxTargets == 1 && greediness >= 0.9f) {
                    scanStepX = 1;
                    // P41 belt track: coarser L1 Y stride (Halcon §2.4.1 motion ROI).
                    scanStepY = (m_hasFindHint && m_trackLastFind) ? 6 : 4;
                } else {
                    scanStepX = scanStepY = 1 << std::min(2, topLevel);
                    if (topLevel == 1 && greediness >= 0.85f)
                        scanStepX = scanStepY = std::max(scanStepX, coarseScenePixels > 900 * 900 ? 4 : 3);
                }
            } else if (topLevel > 0 && coarsePass == 0 && smallScene && maxTargets == 1) {
                // ROI-sized scene at coarse pyramid level (Halcon §2.4.1 search-space limit).
                scanStepX = scanStepY = (coarseScenePixels <= 128 * 128) ? 4 : 3;
            } else if (topLevel == 0 && coarsePass == 0 && smallScene) {
                scanStepX = scanStepY = (coarseScenePixels <= 256 * 256) ? 3 : 2;
            } else if (topLevel == 0 && coarsePass == 0 && coarseScenePixels > 480 * 480) {
                if (elongHorizTpl) {
                    // P11 belt pass 0: Y stride 4 (Halcon §2.4.2 subsampling); pass 1 uses Y-band (§2.4.1).
                    scanStepX = 1;
                    scanStepY = (coarsePass == 0) ? 4 : 3;
                } else if (elongVertTpl) {
                    scanStepX = 3;
                    scanStepY = 2;
                } else {
                    scanStepX = scanStepY =
                        (greediness >= 0.9f && coarseScenePixels > 1200 * 1200) ? 4 : 3;
                }
            } else if (topLevel > 0 && coarsePass > 0) {
                scanStepX = scanStepY = 2;
            } else if (topLevel == 0 && coarsePass > 0) {
                scanStepX = scanStepY = 2;
            }
            if (coarsePass == 0) {
                coarseScanStepXUsed = scanStepX;
                coarseScanStepYUsed = scanStepY;
                coarseScanStepUsed = std::max(scanStepX, scanStepY);
            }

            for (auto& b : coarseBuckets)
                b.clear();

            // P11 Halcon §2.4.1: belt retry pass scans Y-band around pass-0 seeds only.
            int yScanLo = 0;
            int yScanHi = globalYMax;
            if (coarsePass > 0 && elongHorizTpl && largeScene && !beltPassYSeeds.empty()) {
                int y0 = beltPassYSeeds[0], y1 = beltPassYSeeds[0];
                for (int ys : beltPassYSeeds) {
                    y0 = std::min(y0, ys);
                    y1 = std::max(y1, ys);
                }
                const int yBand = std::max(tplH * 2, 36);
                yScanLo = std::max(0, y0 - yBand);
                yScanHi = std::min(globalYMax, y1 + yBand);
            }

            int xMaxAll = 0;
            for (const CoarseModelCtx& c : ctxs)
                xMaxAll = std::max(xMaxAll, c.xMax);

            // P13/P16 belt: top-K X peaks per row at coarse pyramid top (Halcon §3.2.6).
            const bool beltMultiPeakRow = elongHorizTpl && largeScene
                && maxTargets == 1 && greediness >= 0.9f;
            const int beltTopK = beltTrackFast ? 3 : 4;
            const int beltXScanStep = (topLevel > 0)
                ? ((m_hasFindHint && m_trackLastFind) ? 3 : 2)
                : (beltTrackFast ? 5 : 3);
            const bool beltCoarse3Angle = beltMultiPeakRow && topLevel == 0 && nLevels <= 1
                && ctxs.size() >= 4 && (m_maxAngle - m_minAngle) <= 20.5 + 1e-3;
            const float beltCoarseAngleLim = (float)m_effectiveAngleStep + 0.1f;
            const int coarseFeatStride = (maxTargets == 1 && greediness >= 0.9f)
                ? ((beltMultiPeakRow) ? 2 : ((smallScene && topLevel > 0) ? 2 : 1))
                : 1;

            std::vector<CoarseModelCtx> coarseScanCtxs;
            const CoarseModelCtx* coarseScanPtr = ctxs.data();
            int coarseScanN = (int)ctxs.size();
            if (beltCoarse3Angle) {
                coarseScanCtxs.reserve(ctxs.size());
                for (const CoarseModelCtx& c : ctxs) {
                    if (std::fabs(m_models[c.mi].angle) <= beltCoarseAngleLim)
                        coarseScanCtxs.push_back(c);
                }
                if (!coarseScanCtxs.empty()) {
                    coarseScanPtr = coarseScanCtxs.data();
                    coarseScanN = (int)coarseScanCtxs.size();
                }
            }
            const int imgColsTop = topNx.cols;
            const int imgRowsTop = topNx.rows;
            auto maskAtTop = [&](int, int px, int py, const ShapeModelLevel& ml) -> bool {
                return maskAllowed(topLevel, px, py, ml);
            };

            auto scoreCoarseFused = [&](int x, int yPos, int& outMi, float& outS,
                                        int featStride) -> bool {
                outMi = -1;
                outS = -1.f;
                for (int ci = 0; ci < coarseScanN; ++ci) {
                    const CoarseModelCtx& c = coarseScanPtr[ci];
                    if (yPos > c.yMax || x > c.xMax) continue;
                    if (!maskAtTop(0, x, yPos, *c.ml)) continue;
                    const int nEff = (c.n + featStride - 1) / featStride;
                    float accPos = 0.f, accNeg = 0.f;
                    if (!coarseScoreAccumulate(
                            x, yPos, c.n, c.feats, c.ox, c.oy,
                            nxBase, nyBase, mgBase, stepNx, stepMg,
                            coarseMagFloorTop, metric, c.thr, passGreediness,
                            accPos, accNeg, featStride, imgColsTop, imgRowsTop))
                        continue;
                    const float s = metricScore(accPos, accNeg, nEff, metric);
                    if (s >= c.thr && s > outS) { outS = s; outMi = c.mi; }
                }
                return outMi >= 0;
            };

#pragma omp parallel
            {
#ifdef _OPENMP
                const int tid = omp_get_thread_num();
#else
                const int tid = 0;
#endif
                auto& local = coarseBuckets[(size_t)tid];
                if (local.capacity() < 64u)
                    local.reserve(64);

#pragma omp for schedule(static, 4) nowait
                for (int y = yScanLo; y <= yScanHi; y += scanStepY) {
                    if ((y & 31) == 0 && timer.expired.load(std::memory_order_relaxed)) continue;
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
                    if (elongHorizTpl && scanStepX == 1 && y + scanStepY <= globalYMax) {
                        const int py = y + scanStepY;
                        _mm_prefetch((const char*)(nxBase + py * stepNx), _MM_HINT_T0);
                        _mm_prefetch((const char*)(mgBase + py * stepMg), _MM_HINT_T0);
                    }
#elif defined(__GNUC__) || defined(__clang__)
                    if (elongHorizTpl && scanStepX == 1 && y + scanStepY <= globalYMax) {
                        const int py = y + scanStepY;
                        __builtin_prefetch(nxBase + py * stepNx, 0, 1);
                        __builtin_prefetch(mgBase + py * stepMg, 0, 1);
                    }
#endif

                    auto tryCoarseFused = [&](int x) {
                        int bestMi = -1;
                        float bestS = -1.f;
                        if (!scoreCoarseFused(x, y, bestMi, bestS, coarseFeatStride)) return;
                        RawCand rc;
                        rc.modelIdx = bestMi; rc.x = x; rc.y = y; rc.score = bestS;
                        local.push_back(rc);
                    };

                    // P9/P10/P13 belt NumMatches=1: top-K X peaks per Y row with period buckets
                    // (Halcon §3.2.6 / PatMax memory array). P13: X stride-3 + featStride-2 scan,
                    // local refine stride-1 (Halcon §2.4.2 / Cognex OptimizeFindPatMaxPatterns).
                    if (beltMultiPeakRow) {
                        const int bucketW = std::max(8, beltWTop / 3);
                        std::unordered_map<int, RawCand> xBest;
                        xBest.reserve((size_t)beltTopK * 2);
                        for (int x = 0; x <= xMaxAll; x += beltXScanStep) {
                            int bmi = -1;
                            float bs = -1.f;
                            if (!scoreCoarseFused(x, y, bmi, bs, coarseFeatStride)) continue;
                            const int bx = x / bucketW;
                            auto it = xBest.find(bx);
                            if (it == xBest.end() || bs > it->second.score) {
                                RawCand rc;
                                rc.modelIdx = bmi;
                                rc.x = x;
                                rc.y = y;
                                rc.score = bs;
                                xBest[bx] = rc;
                            }
                        }
                        std::vector<RawCand> rowPeaks;
                        rowPeaks.reserve(xBest.size());
                        for (const auto& kv : xBest)
                            rowPeaks.push_back(kv.second);
                        if ((int)rowPeaks.size() > beltTopK) {
                            std::partial_sort(rowPeaks.begin(), rowPeaks.begin() + beltTopK,
                                rowPeaks.end(),
                                [](const RawCand& a, const RawCand& b) { return a.score > b.score; });
                            rowPeaks.resize((size_t)beltTopK);
                        }
                        // Halcon §2.4.4 near-top: resurrect alias peaks trimmed by top-K.
                        const float nearRowTop = rowPeaks.empty() ? 0.f
                            : rowPeaks[0].score - nearTopResurrect;
                        for (const auto& kv : xBest) {
                            const RawCand& c = kv.second;
                            if (c.score < nearRowTop) continue;
                            bool dup = false;
                            for (const RawCand& kept : rowPeaks) {
                                if (kept.modelIdx != c.modelIdx) continue;
                                const int dx = c.x - kept.x;
                                if (dx * dx < bucketW * bucketW) { dup = true; break; }
                            }
                            if (!dup)
                                rowPeaks.push_back(c);
                        }
                        // PatMax stride+local refine on final row peaks only (featStride=1).
                        const int refineRx = beltXScanStep * 3;
                        for (RawCand& rc : rowPeaks) {
                            const int xLo = std::max(0, rc.x - refineRx);
                            const int xHi = std::min(xMaxAll, rc.x + refineRx);
                            for (int xr = xLo; xr <= xHi; ++xr) {
                                int bmi = -1;
                                float bs = -1.f;
                                if (!scoreCoarseFused(xr, y, bmi, bs, 1)) continue;
                                if (bs > rc.score) {
                                    rc.score = bs;
                                    rc.x = xr;
                                    rc.modelIdx = bmi;
                                }
                            }
                        }
                        for (const RawCand& rc : rowPeaks)
                            local.push_back(rc);
                    } else if (scanStepX == 1 && elongHorizTpl && largeScene) {
                        int x = 0;
                        for (; x + 3 <= xMaxAll; x += 4) {
                            tryCoarseFused(x);
                            tryCoarseFused(x + 1);
                            tryCoarseFused(x + 2);
                            tryCoarseFused(x + 3);
                        }
                        for (; x <= xMaxAll; ++x)
                            tryCoarseFused(x);
                    } else {
                        for (int x = 0; x <= xMaxAll; x += scanStepX)
                            tryCoarseFused(x);
                    }
                }
            }

            for (const auto& b : coarseBuckets)
                topCands.insert(topCands.end(), b.begin(), b.end());

            if (!topCands.empty()) {
                float bestCoarse = 0.f;
                for (const RawCand& c : topCands)
                    bestCoarse = std::max(bestCoarse, c.score);
                // NumMatches=1: strong Accept peak — skip retry passes (Halcon §2.4.4).
                // Belt: never early-break — alias peak can dominate pass 0 (130530 NumMatches=1 miss).
                if (maxTargets == 1 && greediness >= 0.9f && !elongHorizTpl
                    && bestCoarse >= coarseSearchScore * 0.88f)
                    break;
                if (beltTrackFast && bestCoarse >= coarseSearchScore * 0.85f)
                    break;
                // Strong coarse peak vs Accept (not MinScore) — avoid early exit that blocks fine refine.
                if (bestCoarse >= coarseThrUsed + 0.03f)
                    break;
                // P11/P12: save unique Y seeds before pass-1 clear (Halcon §2.4.1 search-space limit).
                if (coarsePass == 0 && elongHorizTpl && largeScene && !topCands.empty()) {
                    beltPassYSeeds.clear();
                    std::vector<RawCand> ySorted = topCands;
                    if (ySorted.size() > 32) {
                        std::partial_sort(ySorted.begin(), ySorted.begin() + 32, ySorted.end(),
                            [](const RawCand& a, const RawCand& b) { return a.score > b.score; });
                        ySorted.resize(32);
                    }
                    for (const RawCand& c : ySorted)
                        beltPassYSeeds.push_back(c.y);
                    std::sort(beltPassYSeeds.begin(), beltPassYSeeds.end());
                    beltPassYSeeds.erase(
                        std::unique(beltPassYSeeds.begin(), beltPassYSeeds.end()),
                        beltPassYSeeds.end());
                    if (!beltPassYSeeds.empty()) {
                        const int yBand = std::max(tplH * 2, 36);
                        const int y0 = beltPassYSeeds.front();
                        const int y1 = beltPassYSeeds.back();
                        beltMotionYLo = std::max(0, y0 - yBand);
                        beltMotionYHi = std::min(globalYMax, y1 + yBand);
                        haveBeltMotionY = true;
                    }
                }
                // Weak / spurious coarse hits — lower Accept and rescan (Halcon internal retry).
                if (coarsePass + 1 < maxCoarsePasses)
                    topCands.clear();
            }
        }
    }
    } // !beltPyr2Fast

    // P17/P18/P19: pyramid=2 belt — L0 coarse (P13 path); grad built here once (Halcon §3.1 ROI).
    if (beltPyr2Fast) {
        topCands.clear();
        ensureScenePyramid(srcGray, nLevels, 0, sp, true);
        const cv::Mat& l0Nx = sp.nx[0];
        const cv::Mat& l0Ny = sp.ny[0];
        const cv::Mat& l0Mag = sp.mag[0];
        if (!l0Nx.empty()) {
            const float* nxL0 = l0Nx.ptr<float>(0);
            const float* nyL0 = l0Ny.ptr<float>(0);
            const float* mgL0 = l0Mag.ptr<float>(0);
            const int stepNx0 = (int)(l0Nx.step / sizeof(float));
            const int stepMg0 = (int)(l0Mag.step / sizeof(float));
            const float magFloorL0 = coarseMagFloor(magFloor, nLevels, 0);
            const int tplW0 = std::max(1, m_templateMeta.empty() ? m_templateRefW : m_templateMeta[0].refW);
            const int beltW0 = std::max(tplW0, m_models.empty() ? 0 : m_models[0].levels[0].width);
            int maxModelH0 = 0;
            for (const auto& pm : m_models) {
                if (!pm.levels.empty())
                    maxModelH0 = std::max(maxModelH0, pm.levels[0].height);
            }

            // P19/P20 Halcon MinContrast §2.5: row gate — skip Y rows with no edge energy.
            std::vector<float> rowMaxMag;
            sampleRowMaxMag(l0Mag, rowMaxMag);
            auto rowGateOk = [&](int yPos) -> bool {
                if (maxModelH0 <= 0) return true;
                const int yEnd = std::min(yPos + maxModelH0, (int)rowMaxMag.size() - 1);
                float rm = 0.f;
                for (int ry = yPos; ry <= yEnd; ++ry)
                    rm = std::max(rm, rowMaxMag[(size_t)ry]);
                return rm >= magFloorL0 * 0.35f;
            };

            std::vector<CoarseModelCtx> l0Ctxs;
            l0Ctxs.reserve(numModels);
            int globalYMax0 = 0;
            for (int mi = 0; mi < numModels; ++mi) {
                const ShapeModelLevel& ml = m_models[mi].levels[0];
                if (ml.features.empty() || ml.width > l0Nx.cols || ml.height > l0Nx.rows) continue;
                CoarseModelCtx c;
                c.mi = mi;
                c.ml = &ml;
                c.feats = ml.features.data();
                c.ox = ml.featOx.data();
                c.oy = ml.featOy.data();
                c.n = (int)ml.features.size();
                c.yMax = l0Nx.rows - ml.height;
                c.xMax = l0Nx.cols - ml.width;
                c.thr = coarseThrUsed;
                globalYMax0 = std::max(globalYMax0, c.yMax);
                l0Ctxs.push_back(c);
            }
            if (l0Ctxs.size() > 1) {
                std::stable_sort(l0Ctxs.begin(), l0Ctxs.end(), [this](const CoarseModelCtx& a, const CoarseModelCtx& b) {
                    return std::fabs(m_models[a.mi].angle) < std::fabs(m_models[b.mi].angle);
                });
            }

            // P20 PatMax coarse granularity: 3-angle subset (-5°,0°,+5°) pre-filtered (Halcon §2.4).
            std::vector<CoarseModelCtx> l0CoarseCtxs;
            const bool beltCoarse3AngleL0 = l0Ctxs.size() >= 4
                && (m_maxAngle - m_minAngle) <= 20.5 + 1e-3;
            const float beltCoarseAngleLimL0 = (float)m_effectiveAngleStep + 0.1f;
            if (beltCoarse3AngleL0) {
                l0CoarseCtxs.reserve(3);
                for (const CoarseModelCtx& c : l0Ctxs) {
                    if (std::fabs(m_models[c.mi].angle) <= beltCoarseAngleLimL0)
                        l0CoarseCtxs.push_back(c);
                }
            }
            const std::vector<CoarseModelCtx>& scanCtxs =
                beltCoarse3AngleL0 && !l0CoarseCtxs.empty() ? l0CoarseCtxs : l0Ctxs;

            if (!l0Ctxs.empty()) {
                const int xMaxL0 = [&]() {
                    int xm = 0;
                    for (const CoarseModelCtx& c : l0Ctxs) xm = std::max(xm, c.xMax);
                    return xm;
                }();
                const int bucketWL0 = std::max(8, beltW0 / 3);
                const bool useBeltHintScanPre = m_hasFindHint && m_trackLastFind;
                // P33 track: coarser L0 scan in hint band (refine pass still stride=1).
                const int beltXStepL0 = (beltTrackFast || useBeltHintScanPre) ? 6 : 4;
                const int beltYStepL0 = (beltTrackFast || useBeltHintScanPre) ? 6 : 4;
                const int beltTopKL0 = 4;
                const float passG0 = greediness;
                const int scanFeatStride = (beltTrackFast || useBeltHintScanPre) ? 5 : 3;
                const int imgColsL0 = l0Nx.cols;
                const int imgRowsL0 = l0Nx.rows;
                const int numBucketsL0 = std::min(64, xMaxL0 / bucketWL0 + 2);

                int beltScanXLo = 0;
                int beltScanXHi = xMaxL0;
                int beltScanYLo = 0;
                int beltScanYHi = globalYMax0;
                const bool useBeltHintScan = m_hasFindHint && m_trackLastFind && !l0Ctxs.empty();
                if (useBeltHintScan) {
                    const ShapeModelLevel& mlRef = *l0Ctxs[0].ml;
                    const float hintPlX = (float)(m_findHintCol - searchOffset.x - mlRef.originX
                        - m_positionBiasX);
                    const float hintPlY = (float)(m_findHintRow - searchOffset.y - mlRef.originY
                        - m_positionBiasY);
                    const int xBand = std::max(tplRefW * 2, 96);
                    const int yBand = std::max(tplRefH * 2, std::max(36, beltYStepL0 * 4));
                    beltScanXLo = std::max(0, (int)std::floor(hintPlX - xBand));
                    beltScanXHi = std::min(xMaxL0, (int)std::ceil(hintPlX + xBand));
                    beltScanYLo = std::max(0, (int)std::floor(hintPlY - yBand));
                    beltScanYHi = std::min(globalYMax0, (int)std::ceil(hintPlY + yBand));
                }

                auto scoreL0 = [&](int x, int yPos, int& outMi, float& outS, int featStride) -> bool {
                    outMi = -1;
                    outS = -1.f;
                    for (const CoarseModelCtx& c : scanCtxs) {
                        if (yPos > c.yMax || x > c.xMax) continue;
                        if (!maskAllowed(0, x, yPos, *c.ml)) continue;
                        const int nEff = (c.n + featStride - 1) / featStride;
                        float accPos = 0.f, accNeg = 0.f;
                        if (!coarseScoreAccumulate(
                                x, yPos, c.n, c.feats, c.ox, c.oy,
                                nxL0, nyL0, mgL0, stepNx0, stepMg0, magFloorL0, metric,
                                c.thr, passG0, accPos, accNeg, featStride,
                                imgColsL0, imgRowsL0))
                            continue;
                        const float s = metricScore(accPos, accNeg, nEff, metric);
                        if (s >= c.thr && s > outS) { outS = s; outMi = c.mi; }
                    }
                    return outMi >= 0;
                };

                auto scanBeltRowL0 = [&](int y, std::vector<RawCand>& outLocal) {
                    if (!rowGateOk(y)) return;
                    std::vector<RawCand> bucketBest((size_t)numBucketsL0);
                    std::vector<uint8_t> bucketHas((size_t)numBucketsL0, 0);
                    const int xRowLo = std::max(0, beltScanXLo);
                    const int xRowHi = std::min(xMaxL0, beltScanXHi);
                    for (int x = xRowLo; x <= xRowHi; x += beltXStepL0) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
                        if (x + beltXStepL0 * 2 <= xRowHi) {
                            const int px = x + beltXStepL0 * 2;
                            _mm_prefetch((const char*)(nxL0 + y * stepNx0 + px), _MM_HINT_T0);
                            _mm_prefetch((const char*)(mgL0 + y * stepMg0 + px), _MM_HINT_T0);
                        }
#endif
                        int bmi = -1;
                        float bs = -1.f;
                        if (!scoreL0(x, y, bmi, bs, scanFeatStride)) continue;
                        int bx = x / bucketWL0;
                        if (bx >= numBucketsL0) bx = numBucketsL0 - 1;
                        if (!bucketHas[(size_t)bx] || bs > bucketBest[(size_t)bx].score) {
                            RawCand rc;
                            rc.modelIdx = bmi; rc.x = x; rc.y = y; rc.score = bs;
                            bucketBest[(size_t)bx] = rc;
                            bucketHas[(size_t)bx] = 1;
                        }
                    }
                    std::vector<RawCand> rowPeaks;
                    rowPeaks.reserve(8);
                    for (int bx = 0; bx < numBucketsL0; ++bx) {
                        if (bucketHas[(size_t)bx])
                            rowPeaks.push_back(bucketBest[(size_t)bx]);
                    }
                    if ((int)rowPeaks.size() > beltTopKL0) {
                        std::partial_sort(rowPeaks.begin(), rowPeaks.begin() + beltTopKL0,
                            rowPeaks.end(),
                            [](const RawCand& a, const RawCand& b) { return a.score > b.score; });
                        rowPeaks.resize((size_t)beltTopKL0);
                    }
                    const float nearRowTop = rowPeaks.empty() ? 0.f
                        : rowPeaks[0].score - nearTopResurrect;
                    for (int bx = 0; bx < numBucketsL0; ++bx) {
                        if (!bucketHas[(size_t)bx]) continue;
                        const RawCand& c = bucketBest[(size_t)bx];
                        if (c.score < nearRowTop) continue;
                        bool dup = false;
                        for (const RawCand& kept : rowPeaks) {
                            if (kept.modelIdx != c.modelIdx) continue;
                            const int dx = c.x - kept.x;
                            if (dx * dx < bucketWL0 * bucketWL0) { dup = true; break; }
                        }
                        if (!dup)
                            rowPeaks.push_back(c);
                    }
                    const int refineRx = beltXStepL0 * 4;
                    for (RawCand& rc : rowPeaks) {
                        const int xLo = std::max(0, rc.x - refineRx);
                        const int xHi = std::min(xMaxL0, rc.x + refineRx);
                        for (int xr = xLo; xr <= xHi; ++xr) {
                            int bmi = -1;
                            float bs = -1.f;
                            if (!scoreL0(xr, y, bmi, bs, 1)) continue;
                            if (bs > rc.score) {
                                rc.score = bs; rc.x = xr; rc.modelIdx = bmi;
                            }
                        }
                    }
                    for (const RawCand& rc : rowPeaks)
                        outLocal.push_back(rc);
                };

                auto runBeltL0Scan = [&](int xLo, int xHi, int yLo, int yHi,
                                         std::vector<RawCand>& out) {
                    const int prevXLo = beltScanXLo;
                    const int prevXHi = beltScanXHi;
                    beltScanXLo = xLo;
                    beltScanXHi = xHi;
                    const int yStart = std::max(0, yLo);
                    const int yEnd = std::min(globalYMax0, yHi);
                    out.clear();
                    out.reserve(384);
#ifdef _OPENMP
#pragma omp parallel
                    {
                        std::vector<RawCand> local;
                        local.reserve(64);
#pragma omp for schedule(static, 16) nowait
                        for (int y = yStart; y <= yEnd; y += beltYStepL0) {
                            if ((y & 31) == 0 && timer.expired.load(std::memory_order_relaxed)) continue;
                            scanBeltRowL0(y, local);
                        }
#pragma omp critical
                        { out.insert(out.end(), local.begin(), local.end()); }
                    }
#else
                    for (int y = yStart; y <= yEnd; y += beltYStepL0) {
                        if ((y & 31) == 0 && timer.expired.load(std::memory_order_relaxed)) continue;
                        scanBeltRowL0(y, out);
                    }
#endif
                    beltScanXLo = prevXLo;
                    beltScanXHi = prevXHi;
                };

                std::vector<RawCand> l0Cands;
                runBeltL0Scan(beltScanXLo, beltScanXHi, beltScanYLo, beltScanYHi, l0Cands);
                if (useBeltHintScan) {
                    float hintBest = 0.f;
                    for (const RawCand& c : l0Cands)
                        hintBest = std::max(hintBest, c.score);
                    // P43: stale/wrong-stripe hint — expand when band weak or prior hit unreliable.
                    const float hintExpandThr = std::max(
                        coarseSearchScore * 0.40f, userScore01 * 0.82f);
                    const bool weakHintBand = l0Cands.empty()
                        || hintBest + 1e-3f < hintExpandThr
                        || !hintReliable;
                    if (weakHintBand) {
                        std::vector<RawCand> fullXCands;
                        runBeltL0Scan(0, xMaxL0, beltScanYLo, beltScanYHi, fullXCands);
                        if (!fullXCands.empty())
                            l0Cands = std::move(fullXCands);
                        hintBest = 0.f;
                        for (const RawCand& c : l0Cands)
                            hintBest = std::max(hintBest, c.score);
                    }
                    if (l0Cands.empty() || hintBest + 1e-3f < hintExpandThr) {
                        std::vector<RawCand> fullCands;
                        runBeltL0Scan(0, xMaxL0, 0, globalYMax0, fullCands);
                        if (!fullCands.empty())
                            l0Cands = std::move(fullCands);
                    }
                }
                if (!l0Cands.empty()) {
                    {
                        std::vector<RawCand> topX = l0Cands;
                        const size_t kMed = std::min(topX.size(), (size_t)20);
                        if (topX.size() > kMed) {
                            std::partial_sort(topX.begin(), topX.begin() + kMed, topX.end(),
                                [](const RawCand& a, const RawCand& b) { return a.score > b.score; });
                            topX.resize(kMed);
                        }
                        std::vector<int> xs;
                        xs.reserve(topX.size());
                        for (const RawCand& c : topX)
                            xs.push_back(c.x);
                        if (!xs.empty()) {
                            std::sort(xs.begin(), xs.end());
                            beltMedianPlX = (float)xs[xs.size() / 2];
                        }
                    }
                    topCands = std::move(l0Cands);
                    fineTopLevel = 0;
                    beltCoarseAtL0 = true;
                    coarseScanStepUsed = coarseScanStepXUsed = beltXStepL0;
                    coarseScanStepYUsed = beltYStepL0;
                    haveBeltMotionY = false;
                    ctxs = std::move(l0Ctxs);
                    globalYMax = globalYMax0;
                }
            }
        }
    }

    const bool beltFindAtL0 = (nLevels <= 1) || beltCoarseAtL0;

    // P28 Halcon find NumLevels [N,L]: lowest pyramid level to refine (0=L0).
    int findLowestLevel = std::max(0, std::min(m_findLowestPyramidLevel, nLevels - 1));
    if (m_findLowestPyramidLevel == 0 && m_trackLastFind && m_hasFindHint
        && elongHorizTpl && largeScene && maxTargets == 1 && greediness >= 0.9f
        && nLevels >= 2 && !beltCoarseAtL0)
        findLowestLevel = std::max(findLowestLevel, 1);
    const bool skipL0Rescue = (findLowestLevel > 0);
    // P28/P43 belt track: skip deep scan only when prior hint was a reliable hit.
    const bool skipBeltDeepScan = beltPyr2Fast && m_trackLastFind && m_hasFindHint && hintReliable;

    if (topCands.empty()) return emptyOut;

    // Cap weak coarse hits before sort (greediness=0.9 can still flood memory on big scenes).
    const size_t maxCoarsePool = (greediness >= 0.9f)
        ? (smallScene ? 96u
           : (elongHorizTpl && largeScene ? 384u : 256u))
        : 2048u;
    if (topCands.size() > maxCoarsePool) {
        std::partial_sort(topCands.begin(), topCands.begin() + maxCoarsePool, topCands.end(),
            [](const RawCand& a, const RawCand& b) { return a.score > b.score; });
        topCands.resize(maxCoarsePool);
    }

    // Sort by score; optional center tie-break (skip on elongated models over large repetitive scenes).
    ensureScenePyramid(srcGray, nLevels, fineTopLevel, sp, beltCoarseAtL0);
    const cv::Mat& fineRefNx = sp.nx[fineTopLevel];
    const float priorCx = fineRefNx.cols * 0.5f;
    const float priorCy = fineRefNx.rows * 0.5f;
    std::sort(topCands.begin(), topCands.end(),
              [](const RawCand& a, const RawCand& b) { return a.score > b.score; });
    const float coarseTopGap = topCands.size() >= 2
        ? (topCands[0].score - topCands[1].score) : 1.f;
    // P32: track uses motion-hint X; cold uses median coarse X for stripe disambiguation.
    float beltPriorPlX = beltMedianPlX;
    if (m_hasFindHint && m_trackLastFind && elongHorizTpl && !m_models.empty()) {
        const ShapeModelLevel& mlRef = m_models[0].levels[0];
        beltPriorPlX = (float)(m_findHintCol - searchOffset.x - mlRef.originX
            - m_positionBiasX);
    }
    // P26/P32: median-X tie-break when coarse peaks ambiguous (gap widened to 0.12).
    const bool useBeltMedianPrior =
        beltFindAtL0 && elongHorizTpl && beltPriorPlX >= 0.f && coarseTopGap < 0.12f;
    const bool useCenterPrior =
        !useBeltMedianPrior
        && !(beltFindAtL0 && tplAspect >= 3.f
             && coarseScenePixels > (float)(beltWTop * beltHTop) * 4.f);
    if (useBeltMedianPrior || useCenterPrior) {
        std::stable_sort(topCands.begin(), topCands.end(),
                  [priorCx, priorCy, useCenterPrior, useBeltMedianPrior, beltPriorPlX](
                      const RawCand& a, const RawCand& b) {
                      if (std::abs(a.score - b.score) > 0.04f)
                          return a.score > b.score;
                      if (useBeltMedianPrior && std::abs(a.score - b.score) <= 0.06f) {
                          const float da = std::abs(a.x - beltPriorPlX);
                          const float db = std::abs(b.x - beltPriorPlX);
                          if (da != db)
                              return da < db;
                      }
                      if (!useCenterPrior)
                          return a.score > b.score;
                      const float da = (a.x - priorCx) * (a.x - priorCx)
                          + (a.y - priorCy) * (a.y - priorCy);
                      const float db = (b.x - priorCx) * (b.x - priorCx)
                          + (b.y - priorCy) * (b.y - priorCy);
                      return da < db;
                  });
    }

    const bool repetitiveBelt =
        elongatedTpl && coarseScenePixels > 480.f * 480.f;

    int hardCap = (nLevels > 1 && !beltCoarseAtL0)
        ? std::max(maxTargets * (greediness >= 0.85f ? 4 : 16),
                   greediness >= 0.85f ? 4 : 32)
        : ((elongHorizTpl && largeScene)
            ? std::max(finePoolCap + 16, 64)
            : std::max(maxTargets * 12, elongatedTpl ? 48 : 24));

    const int topFeatN = m_models.empty() ? 0 : (int)m_models[0].levels[fineTopLevel].features.size();
    const bool sparseModel = topFeatN > 0 && topFeatN < 24;

    if (greediness >= 0.85f && topCands.size() >= 2) {
        const float gap = topCands[0].score - topCands[1].score;
        // Never shrink refine pool to maxTargets on large scenes (belt / area dataset).
        if (gap >= 0.12f && !sparseModel && !elongatedTpl && !largeScene)
            hardCap = std::min(hardCap, maxTargets);
        else if (gap >= 0.06f && !elongatedTpl && !largeScene)
            hardCap = std::min(hardCap, std::max(maxTargets * 2, 2));
        else
            hardCap = std::max(hardCap, elongatedTpl
                ? (largeScene ? std::max(finePoolCap + 16, 64) : std::min(40, finePoolCap + 8))
                : (largeScene ? std::min(36, finePoolCap + 4)
                              : std::max(maxTargets * (beltFindAtL0 ? 16 : 8), 16)));
    }
    if (sparseModel)
        hardCap = std::max(hardCap, std::max(maxTargets * 4, 8));
    if (repetitiveBelt)
        hardCap = std::max(hardCap, std::max(finePoolCap + 24, 72));
    else if (largeScene)
        hardCap = std::min(std::max(hardCap, finePoolCap), finePoolCap + (maxTargets == 1 ? 12 : 8));
    else
        hardCap = std::max(hardCap, finePoolCap);
    // P36: keep refine depth on large scenes — coarse rank-1 can miss at NumMatches=1.
    if (maxTargets == 1 && greediness >= 0.9f && !(elongHorizTpl && largeScene)
        && !largeScene)
        hardCap = std::min(hardCap, finePoolCap);
    if (maxTargets == 1 && largeScene && numModels > 1)
        hardCap = std::max(hardCap, std::max(finePoolCap, 16));
    if (maxTargets == 1 && largeTemplate && largeScene)
        hardCap = std::max(hardCap, std::max(finePoolCap, 20));

    // Spatial dedup then cap (PatMax memory array — one peak per neighbourhood).
    std::vector<RawCand> deduped;
    deduped.reserve(std::min<size_t>(topCands.size(), (size_t)hardCap));
    const int dedupR = std::max(4, coarseScanStepUsed * 2);
    const int dedupR2 = dedupR * dedupR;

    if (beltFindAtL0 && elongHorizFeat) {
        // Feature bbox is horizontally elongated — period bucket dedup.
        const int bucketW = std::max(8, modelWTop > 0 ? modelWTop / 2 : tplW / 2);
        std::unordered_map<int, RawCand> bestPerBucket;
        bestPerBucket.reserve(std::min(topCands.size(), (size_t)hardCap * 2));
        for (const RawCand& c : topCands) {
            const int bx = c.x / bucketW;
            auto it = bestPerBucket.find(bx);
            if (it == bestPerBucket.end() || c.score > it->second.score)
                bestPerBucket[bx] = c;
        }
        std::vector<RawCand> bucketed;
        bucketed.reserve(bestPerBucket.size());
        for (const auto& kv : bestPerBucket)
            bucketed.push_back(kv.second);
        std::sort(bucketed.begin(), bucketed.end(),
                  [](const RawCand& a, const RawCand& b) { return a.score > b.score; });
        for (size_t i = 0; i < bucketed.size() && (int)deduped.size() < hardCap; ++i)
            deduped.push_back(bucketed[i]);
    } else {
        for (size_t i = 0; i < topCands.size() && (int)deduped.size() < hardCap; ++i) {
            bool tooClose = false;
            for (const RawCand& kept : deduped) {
                if (kept.modelIdx != topCands[i].modelIdx) continue;
                const int dx = topCands[i].x - kept.x;
                const int dy = topCands[i].y - kept.y;
                if (dx * dx + dy * dy < dedupR2) { tooClose = true; break; }
            }
            if (!tooClose)
                deduped.push_back(topCands[i]);
        }
    }

    // Belt large scene: period merge — bucket bests rescue peaks missed by coarse top-K.
    if (beltFindAtL0 && elongHorizTpl && largeScene && deduped.size() > 16) {
        const int bucketW = std::max(8, tplW / 3);
        const int mergeR = std::max(6, bucketW / 2);
        const int mergeR2 = mergeR * mergeR;
        std::unordered_map<int, RawCand> xBest;
        xBest.reserve(deduped.size());
        for (const RawCand& c : deduped) {
            const int bx = c.x / bucketW;
            auto it = xBest.find(bx);
            if (it == xBest.end() || c.score > it->second.score)
                xBest[bx] = c;
        }
        std::vector<RawCand> merged;
        merged.reserve(xBest.size() + 40);
        const size_t topGlobal = std::min(deduped.size(), (size_t)std::max(48, hardCap));
        for (size_t i = 0; i < topGlobal; ++i)
            merged.push_back(deduped[i]);
        for (const auto& kv : xBest) {
            const RawCand& c = kv.second;
            bool tooClose = false;
            for (const RawCand& kept : merged) {
                if (kept.modelIdx != c.modelIdx) continue;
                const int dx = c.x - kept.x;
                const int dy = c.y - kept.y;
                if (dx * dx + dy * dy < mergeR2) { tooClose = true; break; }
            }
            if (!tooClose)
                merged.push_back(c);
        }
        // Resurrect near-top coarse peaks that 2D dedup kept but buckets/top-K dropped.
        if (!deduped.empty()) {
            const float nearTop = deduped[0].score - nearTopResurrect;
            for (const RawCand& c : deduped) {
                if (c.score < nearTop) continue;
                bool tooClose = false;
                for (const RawCand& kept : merged) {
                    if (kept.modelIdx != c.modelIdx) continue;
                    const int dx = c.x - kept.x;
                    const int dy = c.y - kept.y;
                    if (dx * dx + dy * dy < mergeR2) { tooClose = true; break; }
                }
                if (!tooClose)
                    merged.push_back(c);
            }
        }
        std::sort(merged.begin(), merged.end(),
                  [](const RawCand& a, const RawCand& b) { return a.score > b.score; });
        deduped = std::move(merged);
    }

    // Belt: strong dominant peak — trim weak fine cands (NumMatches=1 keeps full pool).
    if (beltFindAtL0 && elongHorizTpl && largeScene && deduped.size() > 32
        && maxTargets > 1) {
        const float gap = deduped[0].score - deduped[1].score;
        if (gap >= 0.14f && deduped[0].score >= 0.25f) {
            const int bucketW = std::max(8, tplW / 3);
            const int mergeR = std::max(6, bucketW / 2);
            const int mergeR2 = mergeR * mergeR;
            std::vector<RawCand> trimmed;
            trimmed.reserve(36);
            const size_t keepK = 24;
            for (size_t i = 0; i < std::min(deduped.size(), keepK); ++i)
                trimmed.push_back(deduped[i]);
            const float nearTop = deduped[0].score - nearTopResurrect;
            for (size_t i = keepK; i < deduped.size(); ++i) {
                const RawCand& c = deduped[i];
                if (c.score < nearTop) continue;
                bool tooClose = false;
                for (const RawCand& kept : trimmed) {
                    if (kept.modelIdx != c.modelIdx) continue;
                    const int dx = c.x - kept.x;
                    const int dy = c.y - kept.y;
                    if (dx * dx + dy * dy < mergeR2) { tooClose = true; break; }
                }
                if (!tooClose)
                    trimmed.push_back(c);
            }
            std::sort(trimmed.begin(), trimmed.end(),
                      [](const RawCand& a, const RawCand& b) { return a.score > b.score; });
            deduped = std::move(trimmed);
        }
    }

    // P24 Halcon fast path: dominant non-belt peak — skip expensive dedupe pre-fine refine.
    const bool skipDedupePreFineRefine =
        (maxTargets == 1 && greediness >= 0.9f
        && !elongHorizTpl && !elongHorizFeat && !beltFindAtL0
        && deduped.size() > 4
        && deduped[0].score >= userScore01 + 0.06f
        && (deduped[0].score - deduped[1].score) >= 0.15f)
        || (skipBeltDeepScan && !deduped.empty()
            && deduped[0].score >= userScore01 + 0.04f
            && (deduped.size() < 2
                || deduped[0].score - deduped[1].score >= 0.10f));

    if (coarseScanStepUsed > 1 && !deduped.empty() && !skipDedupePreFineRefine) {
        const cv::Mat& refineNx = sp.nx[fineTopLevel];
        const cv::Mat& refineNy = sp.ny[fineTopLevel];
        const cv::Mat& refineMag = sp.mag[fineTopLevel];
        const int refineR = std::max(coarseScanStepXUsed, coarseScanStepYUsed)
            + (fineTopLevel == 0 ? (elongHorizTpl ? 14 : 8) : 2);
        const float* nxBase = refineNx.ptr<float>(0);
        const float* nyBase = refineNy.ptr<float>(0);
        const float* mgBase = refineMag.ptr<float>(0);
        const int stepNxR = (int)(refineNx.step / sizeof(float));
        const int stepMgR = (int)(refineMag.step / sizeof(float));
        for (RawCand& rc : deduped) {
            const ShapeModelLevel& ml = m_models[rc.modelIdx].levels[fineTopLevel];
            if (ml.features.empty()) continue;
            const int yMax = refineNx.rows - ml.height;
            const int xMax = refineNx.cols - ml.width;
            const int n = (int)ml.features.size();
            const float thr = coarseThrUsed;
            int bestX = rc.x, bestY = rc.y;
            float bestS = rc.score;
            const int yLo = std::max(0, rc.y - refineR);
            const int yHi = std::min(yMax, rc.y + refineR);
            const int xLo = std::max(0, rc.x - refineR);
            const int xHi = std::min(xMax, rc.x + refineR);
            for (int y = yLo; y <= yHi; ++y) {
                for (int x = xLo; x <= xHi; ++x) {
                    if (!maskAllowed(fineTopLevel, x, y, ml)) continue;
                    float accPos = 0.f, accNeg = 0.f;
                    const float magFloorRef = coarseMagFloor(magFloor, nLevels, fineTopLevel);
                    if (!coarseScoreAccumulate(
                            x, y, n, ml.features.data(), ml.featOx.data(), ml.featOy.data(),
                            nxBase, nyBase, mgBase, stepNxR, stepMgR, magFloorRef, metric,
                            thr, 0.f, accPos, accNeg))
                        continue;
                    float s = metricScore(accPos, accNeg, n, metric);
                    if (s > bestS) { bestS = s; bestX = x; bestY = y; }
                }
            }
            rc.x = bestX; rc.y = bestY; rc.score = bestS;
        }
    }

    // P26: never collapse deduped on belt — coarse rank-1 can be alias (130530 @55_53, mt=1 miss).
    if (deduped.size() > 1 && nLevels >= 2 && !sparseModel
        && !(beltFindAtL0 && elongHorizTpl))
    {
        const float gap = deduped[0].score - deduped[1].score;
        if (gap >= 0.20f && deduped[0].score >= userScore01 * 0.85f)
            deduped.resize(1);
        else if (maxTargets == 1 && gap >= 0.08f && deduped[0].score >= userScore01)
            deduped.resize(1);
    }
    // Non-belt pyramid=1: strong dominant coarse peak — refine only top-K (Halcon fast path).
    if (nLevels <= 1 && !elongHorizTpl && !elongVertTpl && !largeScene && deduped.size() > 12
        && deduped[0].score - deduped[1].score >= 0.18f
        && deduped[0].score >= 0.22f)
        deduped.resize(12);

    // Halcon NumMatches — cap pool entering fine stage; resurrect near-top (PatMax Accept safety).
    if ((int)deduped.size() > finePoolCap) {
        const int bucketW = std::max(8, elongHorizTpl ? tplW / 3 : 8);
        const int mergeR = std::max(6, bucketW / 2);
        const int mergeR2 = mergeR * mergeR;
        std::vector<RawCand> capped;
        capped.reserve(finePoolCap + 12);
        for (size_t i = 0; i < deduped.size() && (int)capped.size() < finePoolCap; ++i)
            capped.push_back(deduped[i]);
        const float nearTop = deduped[0].score - nearTopResurrect;
        for (size_t i = capped.size(); i < deduped.size(); ++i) {
            const RawCand& c = deduped[i];
            if (c.score < nearTop) continue;
            bool tooClose = false;
            for (const RawCand& kept : capped) {
                if (kept.modelIdx != c.modelIdx) continue;
                const int dx = c.x - kept.x;
                const int dy = c.y - kept.y;
                if (dx * dx + dy * dy < mergeR2) { tooClose = true; break; }
            }
            if (!tooClose)
                capped.push_back(c);
        }
        std::sort(capped.begin(), capped.end(),
                  [](const RawCand& a, const RawCand& b) { return a.score > b.score; });
        deduped = std::move(capped);
    }

    // Small scene + NumMatches=1: dominant peak fast path (Halcon §2.4.4).
    if (smallScene && maxTargets == 1 && deduped.size() > 1) {
        const float gap = deduped[0].score - deduped[1].score;
        if (gap >= 0.08f && deduped[0].score >= userScore01 * 0.45f)
            deduped.resize(1);
    }
    // NumMatches=1 on non-belt small scenes: trim when top peak dominates AND already strong.
    // P40: never collapse on large scenes — coarse rank-1 ≠ fine best (083933 mt=1 miss).
    if (maxTargets == 1 && greediness >= 0.85f && deduped.size() > 1 && !elongHorizTpl
        && !largeScene) {
        const float gap = deduped[0].score - deduped[1].score;
        if (gap >= 0.08f && deduped[0].score >= userScore01
            && deduped[0].score >= coarseSearchScore * 0.85f)
            deduped.resize(1);
    }
    // Belt NumMatches=1: do NOT collapse deduped — coarse row order ≠ fine score (154231 miss).

    // Shared L0 gradient patch for pyramid=1 multi-candidate fine (one Sobel vs many).
    cv::Mat sharedNx0, sharedNy0, sharedMag0;
    cv::Point sharedOff0(0, 0);
    bool haveSharedL0 = false;
    // P21 Halcon §3.1: belt pyramid=2 already built full L0 grad for coarse — reuse in fine.
    if (fineTopLevel == 0 && beltCoarseAtL0 && beltPyr2Fast && !sp.nx[0].empty()) {
        sharedNx0 = sp.nx[0];
        sharedNy0 = sp.ny[0];
        sharedMag0 = sp.mag[0];
        sharedOff0 = cv::Point(0, 0);
        haveSharedL0 = true;
    }
    if (fineTopLevel == 0 && deduped.size() > 1 && !haveSharedL0) {
        int maxW0 = 0, maxH0 = 0;
        for (int mi = 0; mi < numModels; ++mi) {
            const ShapeModelLevel& ml0 = m_models[mi].levels[0];
            maxW0 = std::max(maxW0, ml0.width);
            maxH0 = std::max(maxH0, ml0.height);
        }
        const int pad = std::max(coarseScanStepUsed + 14, 18);
        cv::Rect merged;
        if (beltFindAtL0 && elongHorizTpl && largeScene) {
            // Belt strip: Y band from top coarse + near-top only (ignore spatial Y outliers).
            int y0 = std::numeric_limits<int>::max(), y1 = 0;
            const float nearTopY = deduped.empty() ? 0.f : deduped[0].score - nearTopResurrect;
            const size_t topYCount = std::min(deduped.size(), (size_t)28);
            for (size_t i = 0; i < deduped.size(); ++i) {
                if (i >= topYCount && deduped[i].score < nearTopY) continue;
                y0 = std::min(y0, deduped[i].y - pad);
                y1 = std::max(y1, deduped[i].y + maxH0 + pad);
            }
            if (y0 <= y1) {
                y0 = std::max(0, y0);
                y1 = std::min(srcGray.rows, y1);
                // P12: tighten strip to coarse motion Y-band (Halcon §2.4.1 tracking ROI).
                if (haveBeltMotionY) {
                    y0 = std::max(y0, beltMotionYLo);
                    y1 = std::min(y1, beltMotionYHi + maxH0 + pad);
                }
                if (y0 < y1)
                    merged = cv::Rect(0, y0, srcGray.cols, y1 - y0);
                else
                    merged = cv::Rect(0, 0, srcGray.cols, srcGray.rows);
            } else {
                merged = cv::Rect(0, 0, srcGray.cols, srcGray.rows);
            }
        } else {
            int x0 = std::numeric_limits<int>::max(), y0 = std::numeric_limits<int>::max(), x1 = 0, y1 = 0;
            for (const RawCand& rc : deduped) {
                x0 = std::min(x0, rc.x - pad);
                y0 = std::min(y0, rc.y - pad);
                x1 = std::max(x1, rc.x + maxW0 + pad);
                y1 = std::max(y1, rc.y + maxH0 + pad);
            }
            merged = cv::Rect(x0, y0, x1 - x0, y1 - y0);
        }
        merged &= cv::Rect(0, 0, srcGray.cols, srcGray.rows);
        if (merged.width > 0 && merged.height > 0) {
            bool useShared = false;
            if (beltFindAtL0 && elongHorizTpl && largeScene) {
                const size_t stripArea = (size_t)merged.area();
                const int estPad = pad + 8;
                const size_t perRoi = (size_t)(estPad * 2 + maxW0) * (size_t)(estPad * 2 + maxH0);
                const size_t totalSerial = perRoi * deduped.size();
                useShared = stripArea < totalSerial;
            } else {
                useShared = (size_t)merged.area() < (size_t)coarseScenePixels / 2;
            }
            if (useShared) {
                computeGradLevel(srcGray(merged), sharedNx0, sharedNy0, sharedMag0);
                sharedOff0 = merged.tl();
                haveSharedL0 = true;
            }
        }
    }

    // P23 Halcon §2.4.4 / Cognex PatMax Accept: belt repetitive texture — coarse featStride
    // can rank the wrong stripe; re-rank ambiguous top-K with full L0 Halcon score.
    if (beltFindAtL0 && elongHorizTpl && largeScene && maxTargets == 1
        && deduped.size() >= 2) {
        const cv::Mat* rrNx = nullptr;
        const cv::Mat* rrNy = nullptr;
        const cv::Mat* rrMag = nullptr;
        cv::Point rrOff(0, 0);
        if (haveSharedL0) {
            rrNx = &sharedNx0; rrNy = &sharedNy0; rrMag = &sharedMag0;
            rrOff = sharedOff0;
        } else if (beltCoarseAtL0 && !sp.nx[0].empty()) {
            rrNx = &sp.nx[0]; rrNy = &sp.ny[0]; rrMag = &sp.mag[0];
        }
        if (rrNx != nullptr) {
            const float topGap = deduped[0].score - deduped[1].score;
            const int reRankK = skipBeltDeepScan
                ? std::min((int)deduped.size(), 6)
                : std::min((int)deduped.size(), 12);
            const bool skipReRank = (beltTrackFast
                    && topGap >= 0.06f && deduped[0].score >= userScore01)
                || (skipBeltDeepScan
                    && topGap >= 0.08f && deduped[0].score >= userScore01 + 0.03f)
                || (topGap >= 0.12f && deduped[0].score >= userScore01 + 0.05f);
            const bool ambiguous = !skipReRank && (topGap < 0.14f
                || (deduped[0].score < userScore01 + 0.10f && reRankK >= 3));
            if (ambiguous) {
                const auto rrSpX = [&](float x) { return x - (float)rrOff.x; };
                const auto rrSpY = [&](float y) { return y - (float)rrOff.y; };
                const int snapRx = std::max(8, coarseScanStepXUsed * 3);
                for (int i = 0; i < reRankK; ++i) {
                    RawCand& rc = deduped[(size_t)i];
                    const ShapeModelLevel& ml = m_models[rc.modelIdx].levels[0];
                    const int xMax = rrNx->cols - ml.width;
                    int bestX = rc.x;
                    float bestS = rc.score;
                    const int xLo = std::max(0, rc.x - snapRx);
                    const int xHi = std::min(xMax, rc.x + snapRx);
                    for (int x = xLo; x <= xHi; ++x) {
                        if (!maskAllowed(0, x, rc.y, ml)) continue;
                        const float sFull = scoreAtHalcon(
                            *rrNx, *rrNy, *rrMag,
                            rrSpX((float)x), rrSpY((float)rc.y),
                            ml.features.data(), (int)ml.features.size(),
                            fineMagFloor, metric);
                        if (sFull > bestS) {
                            bestS = sFull;
                            bestX = x;
                        }
                    }
                    rc.x = bestX;
                    rc.score = bestS;
                }
                std::partial_sort(
                    deduped.begin(), deduped.begin() + reRankK, deduped.end(),
                    [](const RawCand& a, const RawCand& b) { return a.score > b.score; });
                std::sort(deduped.begin() + reRankK, deduped.end(),
                          [](const RawCand& a, const RawCand& b) { return a.score > b.score; });
            }
        }
    }

    // Level 0: local ROI Sobel around each coarse peak (Halcon never materializes full-scene L0 grad).

    // Pre-build intermediate pyramid grads before OMP (avoids critical-section storms).
    for (int l = fineTopLevel - 1; l >= 1; --l) {
        if (sp.nx[l].empty())
            ensureScenePyramid(srcGray, nLevels, l, sp);
    }

    // ===== STAGE 2 (PatMax fine filter): refine each memory-array candidate. =====
    //
    // At each finer level: scale candidate coords *= 2, search +/-localR for the
    // best (modelIdx, x, y) using bilinear gradient sampling. The angle is
    // refined against +/-1 neighbour models at every level.

    struct Refined {
        int modelIdx;
        float x, y;
        float score;
        float angle;
        float scale;
        float scaleR;
        float scaleC;
    };

    std::vector<Refined> refined;
    refined.reserve(deduped.size());

#ifdef _OPENMP
    const int fineThreads = omp_get_max_threads();
#else
    const int fineThreads = 1;
#endif
    std::vector<std::vector<Refined>> fineBuckets((size_t)fineThreads);

    // Belt NumMatches=1: never single-coarse serial — coarse rank ≠ fine rank (154231 miss).
    // P19 Halcon §2.4.4: dominant belt peak → serial fine top-1; else cap pool.
    int fineCandLimit = (int)deduped.size();
    // P36 Halcon §2.4.4: serial fine top-1 only on small/template ROI scenes.
    // Large scenes: coarse rank-1 ≠ fine best (154617 mt=1 miss).
    if (maxTargets == 1 && greediness >= 0.9f && !elongHorizTpl && !largeScene)
        fineCandLimit = 1;
    else if (maxTargets == 1 && greediness >= 0.9f && !elongHorizTpl && largeScene) {
        fineCandLimit = std::min(fineCandLimit, finePoolCap);
        // P41 speed: large template with strong coarse peak — shallow fine pool.
        if (largeTemplate && !deduped.empty()
            && deduped[0].score + 1e-3f >= userScore01 + 0.04f)
            fineCandLimit = std::min(fineCandLimit, 8);
    }
    else if (beltFindAtL0 && elongHorizTpl && maxTargets == 1 && greediness >= 0.9f) {
        // P26: never serial-fine top-1 on belt — coarse rank-1 can be alias stripe (130530 @55_53).
        // P42 track: shallow fine pool when motion hint (area2 speed).
        const int beltFineCap = skipBeltDeepScan ? 8 : (beltTrackFast ? 6 : 16);
        fineCandLimit = std::min(fineCandLimit, std::min(finePoolCap, beltFineCap));
    }
    const bool fineSerial = (fineCandLimit == 1);

    if (fineSerial) {
        std::vector<Refined> local;
        local.reserve(1);
        for (int ci = 0; ci < fineCandLimit; ++ci) {
            if (timer.expired.load(std::memory_order_relaxed)) break;
            const RawCand& start = deduped[ci];
            const float peakRatio = (deduped[0].score > 1e-4f)
                ? start.score / deduped[0].score : 1.f;
            int curX = start.x, curY = start.y, curModel = start.modelIdx;
            float curScore = start.score;
            cv::Mat subNx, subNy, subMag;
            cv::Point subOff(0, 0);
            bool haveSubGrad = false;
            const int fineStart = (coarseStartLevel > 0 && !beltCoarseAtL0) ? (coarseStartLevel - 1) : 0;
            for (int l = fineStart; l >= findLowestLevel; --l) {
                if (coarseStartLevel > 0 && !beltCoarseAtL0) { curX *= 2; curY *= 2; }
                int localR = (greediness >= 0.85f)
                    ? std::min(smallScene ? 8 : 12, std::max(5, 2 + (fineTopLevel - l) * 2))
                    : std::min(12, 4 + (fineTopLevel - l) * 3);
                if (fineTopLevel == 0 && coarseScanStepUsed > 1)
                    localR = std::max(localR, coarseScanStepUsed + 8);
                else if (l == 0 && greediness >= 0.85f)
                    localR = std::max(localR, smallScene ? 6 : (haveSharedL0 ? 12 : 14));
                if (l == 0 && elongHorizTpl && fineTopLevel == 0)
                    localR = std::max(localR, coarseScanStepUsed + 12);
                cv::Mat nx, ny, mag;
                cv::Point gradOff(0, 0);
                bool roiFine = false;
                if (l == 0) {
                    if (haveSharedL0) {
                        nx = sharedNx0; ny = sharedNy0; mag = sharedMag0;
                        gradOff = sharedOff0; roiFine = true;
                    } else {
                        int maxW = 0, maxH = 0;
                        for (int mi = 0; mi < numModels; ++mi) {
                            maxW = std::max(maxW, m_models[mi].levels[0].width);
                            maxH = std::max(maxH, m_models[mi].levels[0].height);
                        }
                        const int pad = localR + 4;
                        cv::Rect roi(curX - pad, curY - pad, maxW + 2 * pad, maxH + 2 * pad);
                        roi &= cv::Rect(0, 0, srcGray.cols, srcGray.rows);
                        if (roi.width <= 0 || roi.height <= 0) { curScore = -2.f; break; }
                        computeGradLevel(srcGray(roi), nx, ny, mag);
                        gradOff = roi.tl(); roiFine = true;
                        subNx = nx; subNy = ny; subMag = mag; subOff = gradOff; haveSubGrad = true;
                    }
                } else {
                    nx = sp.nx[l]; ny = sp.ny[l]; mag = sp.mag[l];
                }
                int candAngles[3] = { curModel, curModel - 1, curModel + 1 };
                if (candAngles[1] < 0)         candAngles[1] = angleWrap ? numModels - 1 : curModel;
                if (candAngles[2] >= numModels) candAngles[2] = angleWrap ? 0 : curModel;
                float bestAlign = -2.f;
                int bestX = curX, bestY = curY, bestM = curModel;
                for (int ai = 0; ai < 3; ++ai) {
                    int mi = candAngles[ai];
                    if (peakRatio >= 0.94f && ai > 0 && l == 0) continue;
                    const ShapeModelLevel& ml = m_models[mi].levels[l];
                    if (ml.features.empty()) continue;
                    int xLo, yLo, xHi, yHi;
                    if (roiFine) {
                        xLo = std::max(0, curX - localR - gradOff.x);
                        yLo = std::max(0, curY - localR - gradOff.y);
                        xHi = std::min(nx.cols - ml.width, curX + localR - gradOff.x);
                        yHi = std::min(ny.rows - ml.height, curY + localR - gradOff.y);
                    } else {
                        xLo = std::max(0, curX - localR); yLo = std::max(0, curY - localR);
                        xHi = std::min(nx.cols - ml.width, curX + localR);
                        yHi = std::min(ny.rows - ml.height, curY + localR);
                    }
                    if (xHi < xLo || yHi < yLo) continue;
                    const float levelMag = (l == 0) ? fineMagFloor : coarseMagFloor(magFloor, nLevels, l);
                    const float levelGate = (l > 0)
                        ? coarseAcceptThreshold((beltFindAtL0) ? coarseSearchScore : userScore01,
                              greediness, nLevels) : fineScanMinAvg;
                    const bool useIntL0 = (l == 0 && roiFine && greediness >= 0.85f);
                    if (useIntL0) {
                        const int stepNxL = (int)(nx.step / sizeof(float));
                        const int stepMgL = (int)(mag.step / sizeof(float));
                        const float* nxBaseL = nx.ptr<float>(0);
                        const float* nyBaseL = ny.ptr<float>(0);
                        const float* mgBaseL = mag.ptr<float>(0);
                        for (int yy = yLo; yy <= yHi; ++yy) {
                            for (int xx = xLo; xx <= xHi; ++xx) {
                                float accPos = 0.f, accNeg = 0.f;
                                if (!coarseScoreAccumulate(xx, yy, (int)ml.features.size(),
                                        ml.features.data(), ml.featOx.data(), ml.featOy.data(),
                                        nxBaseL, nyBaseL, mgBaseL, stepNxL, stepMgL,
                                        levelMag, metric, fineScanMinAvg, greediness, accPos, accNeg))
                                    continue;
                                float s = metricScore(accPos, accNeg, (int)ml.features.size(), metric);
                                if (s > bestAlign) {
                                    bestAlign = s; bestX = xx + gradOff.x; bestY = yy + gradOff.y; bestM = mi;
                                }
                            }
                        }
                    } else {
                        for (int yy = yLo; yy <= yHi; ++yy) {
                            for (int xx = xLo; xx <= xHi; ++xx) {
                                float s = scoreAtAlignmentGreediness(nx, ny, mag, (float)xx, (float)yy,
                                    ml.features.data(), (int)ml.features.size(),
                                    levelMag, metric, levelGate, greediness);
                                if (s < 0.f) continue;
                                if (s > bestAlign) {
                                    bestAlign = s;
                                    bestX = roiFine ? (xx + gradOff.x) : xx;
                                    bestY = roiFine ? (yy + gradOff.y) : yy;
                                    bestM = mi;
                                }
                            }
                        }
                    }
                }
                if (bestAlign < -1.f) { curScore = -2.f; break; }
                const float levelMagVis = (l == 0) ? fineMagFloor : coarseMagFloor(magFloor, nLevels, l);
                const ShapeModelLevel& mlBest = m_models[bestM].levels[l];
                const float scoreX = roiFine ? (float)(bestX - gradOff.x) : (float)bestX;
                const float scoreY = roiFine ? (float)(bestY - gradOff.y) : (float)bestY;
                float levelScore = scoreAtAlignment(nx, ny, mag, scoreX, scoreY,
                    mlBest.features.data(), (int)mlBest.features.size(), levelMagVis, metric);
                const float levelReject = (l > 0)
                    ? coarseAcceptThreshold((nLevels <= 1) ? coarseSearchScore : userScore01,
                          greediness, nLevels) : 0.f;
                if (levelScore < levelReject) { curScore = -2.f; break; }
                curX = bestX; curY = bestY; curModel = bestM; curScore = levelScore;
            }
            if (curScore < -1.f) continue;
            Refined r;
            r.modelIdx = curModel; r.x = (float)curX; r.y = (float)curY; r.score = curScore;
            r.angle = m_models[curModel].angle;
            r.scale = m_models[curModel].scale;
            r.scaleR = m_models[curModel].scaleR; r.scaleC = m_models[curModel].scaleC;
            const ShapeModelLevel& ml0 = m_models[curModel].levels[0];
            const bool skipSubPixelFast = greediness >= 0.9f && curScore >= userScore01
                && (!elongHorizTpl || (beltFindAtL0 && peakRatio >= 0.90f)
                    || (beltTrackFast && curScore >= userScore01 * 0.95f));
            if (m_subPixelMode != ShapeSubPixelMode::None && !ml0.features.empty()
                && !skipSubPixelFast) {
                cv::Mat nx0, ny0, mag0;
                cv::Point off0(0, 0);
                if (haveSubGrad || haveSharedL0) {
                    nx0 = haveSubGrad ? subNx : sharedNx0;
                    ny0 = haveSubGrad ? subNy : sharedNy0;
                    mag0 = haveSubGrad ? subMag : sharedMag0;
                    off0 = haveSubGrad ? subOff : sharedOff0;
                } else {
                    if (sp.nx[0].empty()) ensureScenePyramid(srcGray, nLevels, 0, sp);
                    nx0 = sp.nx[0]; ny0 = sp.ny[0]; mag0 = sp.mag[0];
                }
                const auto spX = [&](float x) { return x - (float)off0.x; };
                const auto spY = [&](float y) { return y - (float)off0.y; };
                float v[3][3];
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                        v[dy + 1][dx + 1] = scoreAtAlignment(nx0, ny0, mag0,
                            spX((float)curX + dx), spY((float)curY + dy),
                            ml0.features.data(), (int)ml0.features.size(), fineMagFloor, metric);
                double ox = 0., oy = 0.;
                if (m_subPixelMode == ShapeSubPixelMode::LeastSquares)
                    quadraticSubpixelOffset2D(v, ox, oy);
                else {
                    ox = parabolicPeakOffset(v[1][0], v[1][1], v[1][2]);
                    oy = parabolicPeakOffset(v[0][1], v[1][1], v[2][1]);
                }
                r.x += (float)ox; r.y += (float)oy;
                r.score = scoreAtHalcon(nx0, ny0, mag0, spX(r.x), spY(r.y),
                    ml0.features.data(), (int)ml0.features.size(), fineMagFloor, metric);
            }
            if (findLowestLevel > 0) {
                for (int up = findLowestLevel - 1; up >= 0; --up) {
                    r.x *= 2.f;
                    r.y *= 2.f;
                }
            }
            if (r.score + 1e-3f >= userScore01)
                local.push_back(r);
        }
        refined.insert(refined.end(), local.begin(), local.end());
    } else {

#pragma omp parallel
    {
#ifdef _OPENMP
        const int fineTid = omp_get_thread_num();
#else
        const int fineTid = 0;
#endif
        auto& local = fineBuckets[(size_t)fineTid];
        if (local.capacity() < deduped.size())
            local.reserve(deduped.size());

#pragma omp for schedule(static, 2) nowait
        for (int ci = 0; ci < fineCandLimit; ++ci) {
            if (timer.expired.load(std::memory_order_relaxed)) continue;
            const RawCand& start = deduped[ci];
            const float peakRatio = (deduped[0].score > 1e-4f)
                ? start.score / deduped[0].score : 1.f;
            int curX = start.x;
            int curY = start.y;
            int curModel = start.modelIdx;
            float curScore = start.score;
            cv::Mat subNx, subNy, subMag;
            cv::Point subOff(0, 0);
            bool haveSubGrad = false;

            // fineTopLevel==0: refine locally at L0 (no coord upscale from coarse pyramid).
            const int fineStart = (coarseStartLevel > 0 && !beltCoarseAtL0) ? (coarseStartLevel - 1) : 0;
            for (int l = fineStart; l >= findLowestLevel; --l) {
                if (coarseStartLevel > 0 && !beltCoarseAtL0) {
                    curX *= 2;
                    curY *= 2;
                }

                int localR = (greediness >= 0.85f)
                    ? std::min(12, std::max(5, 2 + (fineTopLevel - l) * 2))
                    : std::min(12, 4 + (fineTopLevel - l) * 3);
                if (fineTopLevel == 0 && coarseScanStepUsed > 1)
                    localR = std::max(localR, coarseScanStepUsed + 8);
                else if (l == 0 && greediness >= 0.85f)
                    localR = std::max(localR, smallScene ? 8 : (haveSharedL0 ? 12 : 14));
                if (peakRatio < 0.92f && l == 0 && nLevels <= 1)
                    localR = std::max(localR, coarseScanStepUsed + 10);
                else if (l == 0 && peakRatio >= 0.90f && haveSharedL0)
                    localR = std::min(localR, coarseScanStepUsed + 8);
                if (l == 0 && haveSharedL0 && elongHorizTpl && peakRatio >= 0.88f)
                    localR = std::min(localR, coarseScanStepUsed + 7);
                else if (l == 0 && beltFindAtL0 && elongHorizTpl && peakRatio >= 0.90f && !haveSharedL0)
                    localR = std::min(localR, coarseScanStepUsed + 8);

                cv::Mat nx, ny, mag;
                cv::Point gradOff(0, 0);
                bool roiFine = false;
                if (l == 0) {
                    if (haveSharedL0) {
                        nx = sharedNx0;
                        ny = sharedNy0;
                        mag = sharedMag0;
                        gradOff = sharedOff0;
                        roiFine = true;
                    } else {
                        int maxW = 0, maxH = 0;
                        for (int mi = 0; mi < numModels; ++mi) {
                            const ShapeModelLevel& ml0 = m_models[mi].levels[0];
                            maxW = std::max(maxW, ml0.width);
                            maxH = std::max(maxH, ml0.height);
                        }
                        const int pad = localR + 4;
                        cv::Rect roi(curX - pad, curY - pad, maxW + 2 * pad, maxH + 2 * pad);
                        roi &= cv::Rect(0, 0, srcGray.cols, srcGray.rows);
                        if (roi.width <= 0 || roi.height <= 0) {
                            curScore = -2.f;
                            break;
                        }
                        computeGradLevel(srcGray(roi), nx, ny, mag);
                        gradOff = roi.tl();
                        roiFine = true;
                        subNx = nx; subNy = ny; subMag = mag;
                        subOff = gradOff;
                        haveSubGrad = true;
                    }
                } else {
                    nx = sp.nx[l];
                    ny = sp.ny[l];
                    mag = sp.mag[l];
                }

                // Try candidate angles { current, current-1, current+1 }, wrapping
                // for a full 360-degree model set.
                int candAngles[3] = { curModel, curModel - 1, curModel + 1 };
                if (candAngles[1] < 0)         candAngles[1] = angleWrap ? numModels - 1 : curModel;
                if (candAngles[2] >= numModels) candAngles[2] = angleWrap ? 0 : curModel;

                float bestAlign = -2.f;
                int bestX = curX, bestY = curY, bestM = curModel;

                for (int ai = 0; ai < 3; ++ai) {
                    int mi = candAngles[ai];
                    if (peakRatio >= 0.94f && ai > 0 && l == 0 && nLevels <= 1 && !elongHorizTpl)
                        continue;
                    const ShapeModelLevel& ml = m_models[mi].levels[l];
                    if (ml.features.empty()) continue;
                    int xLo, yLo, xHi, yHi;
                    if (roiFine) {
                        xLo = std::max(0, curX - localR - gradOff.x);
                        yLo = std::max(0, curY - localR - gradOff.y);
                        xHi = std::min(nx.cols - ml.width, curX + localR - gradOff.x);
                        yHi = std::min(ny.rows - ml.height, curY + localR - gradOff.y);
                    } else {
                        xLo = std::max(0, curX - localR);
                        yLo = std::max(0, curY - localR);
                        xHi = std::min(nx.cols - ml.width, curX + localR);
                        yHi = std::min(ny.rows - ml.height, curY + localR);
                    }
                    if (xHi < xLo || yHi < yLo) continue;

                    const float levelMag = (l == 0)
                        ? fineMagFloor
                        : coarseMagFloor(magFloor, nLevels, l);
                    const float levelGate = (l > 0)
                        ? coarseAcceptThreshold(
                              (beltFindAtL0) ? coarseSearchScore : userScore01,
                              greediness, nLevels)
                        : fineScanMinAvg;

                    if (l == 0 && roiFine && greediness >= 0.85f) {
                        const int stepNxL = (int)(nx.step / sizeof(float));
                        const int stepMgL = (int)(mag.step / sizeof(float));
                        const float* nxBaseL = nx.ptr<float>(0);
                        const float* nyBaseL = ny.ptr<float>(0);
                        const float* mgBaseL = mag.ptr<float>(0);
                        for (int yy = yLo; yy <= yHi; ++yy) {
                            for (int xx = xLo; xx <= xHi; ++xx) {
                                float accPos = 0.f, accNeg = 0.f;
                                if (!coarseScoreAccumulate(
                                        xx, yy, (int)ml.features.size(),
                                        ml.features.data(), ml.featOx.data(), ml.featOy.data(),
                                        nxBaseL, nyBaseL, mgBaseL,
                                        stepNxL, stepMgL, levelMag, metric,
                                        fineScanMinAvg, greediness, accPos, accNeg))
                                    continue;
                                float s = metricScore(accPos, accNeg, (int)ml.features.size(), metric);
                                if (s > bestAlign) {
                                    bestAlign = s;
                                    bestX = xx + gradOff.x;
                                    bestY = yy + gradOff.y;
                                    bestM = mi;
                                }
                            }
                        }
                    } else {
                        for (int yy = yLo; yy <= yHi; ++yy) {
                            for (int xx = xLo; xx <= xHi; ++xx) {
                                float s = scoreAtAlignmentGreediness(
                                    nx, ny, mag, (float)xx, (float)yy,
                                    ml.features.data(), (int)ml.features.size(),
                                    levelMag, metric, levelGate, greediness);
                                if (s < 0.f) continue;
                                if (s > bestAlign) {
                                    bestAlign = s;
                                    bestX = roiFine ? (xx + gradOff.x) : xx;
                                    bestY = roiFine ? (yy + gradOff.y) : yy;
                                    bestM = mi;
                                }
                            }
                        }
                    }
                }

                if (bestAlign < -1.f) {
                    curScore = -2.f;
                    break;
                }

                const float levelMagVis = (l == 0)
                    ? fineMagFloor
                    : coarseMagFloor(magFloor, nLevels, l);
                const ShapeModelLevel& mlBest = m_models[bestM].levels[l];
                const float scoreX = roiFine ? (float)(bestX - gradOff.x) : (float)bestX;
                const float scoreY = roiFine ? (float)(bestY - gradOff.y) : (float)bestY;
                float levelScore = scoreAtAlignment(
                    nx, ny, mag, scoreX, scoreY,
                    mlBest.features.data(), (int)mlBest.features.size(),
                    levelMagVis, metric);
                const float levelReject = (l > 0)
                    ? coarseAcceptThreshold(
                          (beltFindAtL0) ? coarseSearchScore : userScore01,
                          greediness, nLevels)
                    : 0.f; // defer MinScore until after sub-pixel at L0
                if (levelScore < levelReject) {
                    curScore = -2.f;
                    break;
                }

                curX = bestX; curY = bestY; curModel = bestM; curScore = levelScore;
            }

            if (curScore < -1.f) continue;

            Refined r;
            r.modelIdx = curModel;
            r.x = (float)curX;
            r.y = (float)curY;
            r.score = curScore;
            r.angle = m_models[curModel].angle;
            r.scale = m_models[curModel].scale;
            r.scaleR = m_models[curModel].scaleR;
            r.scaleC = m_models[curModel].scaleC;

            const ShapeModelLevel& ml0 = m_models[curModel].levels[0];

            const bool skipSubPixelFast = maxTargets == 1 && greediness >= 0.9f
                && curScore >= userScore01
                && (!elongHorizTpl || (beltFindAtL0 && peakRatio >= 0.90f)
                    || (beltTrackFast && curScore >= userScore01 * 0.95f)
                    || (skipBeltDeepScan && elongHorizTpl
                        && curScore >= userScore01 - 0.02f && peakRatio >= 0.85f));

            if (m_subPixelMode != ShapeSubPixelMode::None && !ml0.features.empty()
                && !skipSubPixelFast) {
                cv::Mat nx0, ny0, mag0;
                cv::Point off0(0, 0);
                if (haveSubGrad || haveSharedL0) {
                    nx0 = haveSubGrad ? subNx : sharedNx0;
                    ny0 = haveSubGrad ? subNy : sharedNy0;
                    mag0 = haveSubGrad ? subMag : sharedMag0;
                    off0 = haveSubGrad ? subOff : sharedOff0;
                } else {
                    if (sp.nx[0].empty())
                        ensureScenePyramid(srcGray, nLevels, 0, sp);
                    nx0 = sp.nx[0];
                    ny0 = sp.ny[0];
                    mag0 = sp.mag[0];
                }
                const auto spX = [&](float x) { return x - (float)off0.x; };
                const auto spY = [&](float y) { return y - (float)off0.y; };

                // --- sub-pixel position: 2D quadratic over the 3x3 score patch ---
                float v[3][3];
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                        v[dy + 1][dx + 1] = scoreAtAlignment(nx0, ny0, mag0,
                                                    spX((float)curX + dx), spY((float)curY + dy),
                                                    ml0.features.data(),
                                                    (int)ml0.features.size(),
                                                    fineMagFloor, metric);
                double ox = 0., oy = 0.;
                if (m_subPixelMode == ShapeSubPixelMode::LeastSquares)
                    quadraticSubpixelOffset2D(v, ox, oy);
                else {
                    ox = parabolicPeakOffset(v[1][0], v[1][1], v[1][2]);
                    oy = parabolicPeakOffset(v[0][1], v[1][1], v[2][1]);
                }
                r.x += (float)ox;
                r.y += (float)oy;
                r.score = scoreAtHalcon(nx0, ny0, mag0, spX(r.x), spY(r.y),
                                  ml0.features.data(), (int)ml0.features.size(),
                                  fineMagFloor, metric);

                // --- sub-angle: 1D parabola (LeastSquares mode only) ---
                if (m_subPixelMode == ShapeSubPixelMode::LeastSquares) {
                    int mPrev = curModel - 1, mNext = curModel + 1;
                    if (mPrev < 0)          mPrev = angleWrap ? numModels - 1 : -1;
                    if (mNext >= numModels) mNext = angleWrap ? 0 : -1;
                    if (mPrev >= 0 && mNext >= 0) {
                        const ShapeModelLevel& mlP = m_models[mPrev].levels[0];
                        const ShapeModelLevel& mlN = m_models[mNext].levels[0];
                        if (!mlP.features.empty() && !mlN.features.empty()) {
                            float sP = scoreAtAlignment(nx0, ny0, mag0, spX(r.x), spY(r.y),
                                               mlP.features.data(), (int)mlP.features.size(),
                                               fineMagFloor, metric);
                            float sN = scoreAtAlignment(nx0, ny0, mag0, spX(r.x), spY(r.y),
                                               mlN.features.data(), (int)mlN.features.size(),
                                               fineMagFloor, metric);
                            float sC = r.score;
                            double aoff = parabolicPeakOffset(sP, sC, sN);
                            r.angle = (float)(m_models[curModel].angle + aoff * m_effectiveAngleStep);
                            r.score = std::max(sP, std::max(sC, sN));
                        }
                    }
                }
            } else if (m_useIcpRefine) {
                if (sp.nx[0].empty())
                    ensureSourcePyramidLevel(sp, srcGray, nLevels, 0);
                const cv::Mat& nx0  = sp.nx[0];
                const cv::Mat& ny0  = sp.ny[0];
                const cv::Mat& mag0 = sp.mag[0];
                icpRefinePose(nx0, ny0, mag0, r.x, r.y, r.angle, ml0, fineMagFloor, metric);
                r.score = scoreAtHalcon(nx0, ny0, mag0, r.x, r.y,
                                  ml0.features.data(), (int)ml0.features.size(),
                                  fineMagFloor, metric);
            }

            // Borderline rescue (pyramid=1 belt): widen L0 search when score narrowly misses MinScore.
            if (!skipL0Rescue && beltFindAtL0 && !smallScene
                && r.score < userScore01 + 2e-4f
                && r.score >= userScore01 - (elongHorizTpl ? 0.12f : 0.12f))
            {
                cv::Mat nxB, nyB, mgB;
                cv::Point offB(0, 0);
                bool haveBorderGrad = haveSubGrad || haveSharedL0;
                if (haveBorderGrad) {
                    nxB = haveSubGrad ? subNx : sharedNx0;
                    nyB = haveSubGrad ? subNy : sharedNy0;
                    mgB = haveSubGrad ? subMag : sharedMag0;
                    offB = haveSubGrad ? subOff : sharedOff0;
                } else {
                    const int pad = 32;
                    cv::Rect roi((int)std::lround(r.x) - pad, (int)std::lround(r.y) - pad,
                        ml0.width + 2 * pad, ml0.height + 2 * pad);
                    roi &= cv::Rect(0, 0, srcGray.cols, srcGray.rows);
                    if (roi.width > 0 && roi.height > 0) {
                        computeGradLevel(srcGray(roi), nxB, nyB, mgB);
                        offB = roi.tl();
                        haveBorderGrad = true;
                    }
                }
                if (haveBorderGrad) {
                const auto spX = [&](float x) { return x - (float)offB.x; };
                const auto spY = [&](float y) { return y - (float)offB.y; };
                float bestS = r.score;
                float bestX = r.x, bestY = r.y;
                const int stepNxB = (int)(nxB.step / sizeof(float));
                const int stepMgB = (int)(mgB.step / sizeof(float));
                const float* nxBaseB = nxB.ptr<float>(0);
                const float* nyBaseB = nyB.ptr<float>(0);
                const float* mgBaseB = mgB.ptr<float>(0);
                const int rescueR = std::max(coarseScanStepUsed + 10, 14);
                // Belt (Halcon §2.5.6 / Steger): periodic alias along X — widen X, tight Y.
                const int expandRx = elongHorizTpl
                    ? std::min(std::max(tplW * 2, 96), rescueR + 80)
                    : (elongatedTpl ? std::min(std::max(36, tplW / 2), rescueR + 20) : 28);
                const int expandRy = elongHorizTpl
                    ? std::max(12, rescueR + 6)
                    : expandRx;
                const int cx = (int)std::lround(r.x - (float)offB.x);
                const int cy = (int)std::lround(r.y - (float)offB.y);
                const int xLoB = std::max(0, cx - expandRx);
                const int yLoB = std::max(0, cy - expandRy);
                const int xHiB = std::min(nxB.cols - ml0.width, cx + expandRx);
                const int yHiB = std::min(nxB.rows - ml0.height, cy + expandRy);
                int bestMiB = r.modelIdx;
                for (int yy = yLoB; yy <= yHiB; ++yy) {
                    for (int xx = xLoB; xx <= xHiB; ++xx) {
                        for (int miB = 0; miB < numModels; ++miB) {
                            const ShapeModelLevel& mlB = m_models[miB].levels[0];
                            if (mlB.features.empty()) continue;
                            if (xx > nxB.cols - mlB.width || yy > nxB.rows - mlB.height) continue;
                            float accPos = 0.f, accNeg = 0.f;
                            if (!coarseScoreAccumulate(
                                    xx, yy, (int)mlB.features.size(),
                                    mlB.features.data(), mlB.featOx.data(), mlB.featOy.data(),
                                    nxBaseB, nyBaseB, mgBaseB,
                                    stepNxB, stepMgB, fineMagFloor, metric,
                                    0.f, 0.f, accPos, accNeg))
                                continue;
                            float s = metricScore(accPos, accNeg, (int)mlB.features.size(), metric);
                            if (s > bestS) {
                                bestS = s;
                                bestX = (float)xx + (float)offB.x;
                                bestY = (float)yy + (float)offB.y;
                                bestMiB = miB;
                            }
                        }
                    }
                }

                if (bestS > r.score + 1e-4f) {
                    r.x = bestX;
                    r.y = bestY;
                    r.modelIdx = bestMiB;
                    const ShapeModelLevel& mlBest = m_models[r.modelIdx].levels[0];
                    r.score = scoreAtHalcon(nxB, nyB, mgB,
                        spX(r.x), spY(r.y),
                        mlBest.features.data(), (int)mlBest.features.size(),
                        fineMagFloor, metric);
                    if (m_subPixelMode != ShapeSubPixelMode::None) {
                        float v[3][3];
                        for (int dy = -1; dy <= 1; ++dy)
                            for (int dx = -1; dx <= 1; ++dx)
                                v[dy + 1][dx + 1] = scoreAtAlignment(nxB, nyB, mgB,
                                    spX(r.x + dx), spY(r.y + dy),
                                    mlBest.features.data(), (int)mlBest.features.size(),
                                    fineMagFloor, metric);
                        double ox = 0., oy = 0.;
                        if (m_subPixelMode == ShapeSubPixelMode::LeastSquares)
                            quadraticSubpixelOffset2D(v, ox, oy);
                        else {
                            ox = parabolicPeakOffset(v[1][0], v[1][1], v[1][2]);
                            oy = parabolicPeakOffset(v[0][1], v[1][1], v[2][1]);
                        }
                        r.x += (float)ox;
                        r.y += (float)oy;
                        r.score = scoreAtHalcon(nxB, nyB, mgB, spX(r.x), spY(r.y),
                            mlBest.features.data(), (int)mlBest.features.size(),
                            fineMagFloor, metric);
                        if (m_subPixelMode == ShapeSubPixelMode::LeastSquares) {
                            const int curM = r.modelIdx;
                            int mPrev = curM - 1, mNext = curM + 1;
                            if (mPrev < 0)          mPrev = angleWrap ? numModels - 1 : -1;
                            if (mNext >= numModels) mNext = angleWrap ? 0 : -1;
                            if (mPrev >= 0 && mNext >= 0) {
                                const ShapeModelLevel& mlP = m_models[mPrev].levels[0];
                                const ShapeModelLevel& mlN = m_models[mNext].levels[0];
                                if (!mlP.features.empty() && !mlN.features.empty()) {
                                    float sP = scoreAtAlignment(nxB, nyB, mgB, spX(r.x), spY(r.y),
                                        mlP.features.data(), (int)mlP.features.size(),
                                        fineMagFloor, metric);
                                    float sN = scoreAtAlignment(nxB, nyB, mgB, spX(r.x), spY(r.y),
                                        mlN.features.data(), (int)mlN.features.size(),
                                        fineMagFloor, metric);
                                    float sC = r.score;
                                    double aoff = parabolicPeakOffset(sP, sC, sN);
                                    r.angle = (float)(m_models[curM].angle + aoff * m_effectiveAngleStep);
                                    r.score = std::max(sP, std::max(sC, sN));
                                }
                            }
                        }
                    }
                }
                }
            }

            if (findLowestLevel > 0) {
                for (int up = findLowestLevel - 1; up >= 0; --up) {
                    r.x *= 2.f;
                    r.y *= 2.f;
                }
            }
            if (r.score + 1e-3f < userScore01) continue;

            local.push_back(r);
        }
    }

    for (const auto& fb : fineBuckets)
        refined.insert(refined.end(), fb.begin(), fb.end());

    } // !fineSerial

    // P7 belt global fallback (Halcon §2.4.4 / Steger coarse-to-fine): when fine pool misses
    // borderline on repetitive belt, widen L0 search from top coarse seeds (all angles).
    if (!skipL0Rescue && maxTargets == 1 && greediness >= 0.9f && beltFindAtL0
        && elongHorizTpl && largeScene && !deduped.empty())
    {
        float bestRefined = -1.f;
        for (const Refined& rf : refined)
            bestRefined = std::max(bestRefined, rf.score);
        const bool needFallback = refined.empty() || bestRefined < userScore01;
        if (needFallback) {
            cv::Mat fbNx, fbNy, fbMag;
            cv::Point fbOff(0, 0);
            bool haveFbGrad = false;
            if (haveSharedL0) {
                fbNx = sharedNx0; fbNy = sharedNy0; fbMag = sharedMag0;
                fbOff = sharedOff0;
                haveFbGrad = true;
            }
            const int expandRxF = std::min(std::max(tplW * 2, 96), 160);
            const int expandRyF = std::max(20, coarseScanStepUsed + 12);
            const size_t seedK = skipBeltDeepScan
                ? std::min(deduped.size(), (size_t)12)
                : std::min(deduped.size(), (size_t)48);
            Refined bestFb;
            bestFb.score = bestRefined;
            const float borderlineLo = userScore01 - 0.10f;

            // P29 Halcon §2.4.1: belt track — row-track only near motion hint Y-band.
            const bool useHintYBand = skipBeltDeepScan
                || (m_hasFindHint && m_trackLastFind && elongHorizTpl && largeScene);
            int hintPlY = -1, yBandLo = 0, yBandHi = globalYMax;
            if (useHintYBand && m_hasFindHint && !ctxs.empty()) {
                const ShapeModelLevel& mlRef = *ctxs[0].ml;
                hintPlY = (int)std::lround(m_findHintRow - searchOffset.y
                    - mlRef.originY - m_positionBiasY);
                const int yBand = std::max(tplH * 2, std::max(36, coarseScanStepYUsed * 4));
                yBandLo = std::max(0, hintPlY - yBand);
                yBandHi = std::min(globalYMax, hintPlY + yBand);
            }

            // P30 track: coarse peak already >= MinScore — one L0 verify, skip row-track.
            bool trackCoarsePromoted = false;
            if (skipBeltDeepScan && !deduped.empty()
                && deduped[0].score + 1e-3f >= userScore01)
            {
                if (!haveFbGrad) {
                    if (haveSharedL0) {
                        fbNx = sharedNx0; fbNy = sharedNy0; fbMag = sharedMag0;
                        haveFbGrad = true;
                    } else {
                        computeGradLevel(srcGray, fbNx, fbNy, fbMag);
                        haveFbGrad = true;
                    }
                }
                const RawCand& seed = deduped[0];
                const ShapeModelLevel& mlP = m_models[seed.modelIdx].levels[0];
                if (!mlP.features.empty() && !fbNx.empty()) {
                    int px = seed.x, py = seed.y;
                    if (findLowestLevel > 0) {
                        const int up = beltCoarseAtL0 ? 2 : (1 << findLowestLevel);
                        px *= up;
                        py *= up;
                    }
                    const float s0 = scoreAtHalcon(fbNx, fbNy, fbMag,
                        (float)px, (float)py,
                        mlP.features.data(), (int)mlP.features.size(),
                        fineMagFloor, metric);
                    if (s0 + 1e-3f >= userScore01) {
                        bestFb.modelIdx = seed.modelIdx;
                        bestFb.x = (float)px;
                        bestFb.y = (float)py;
                        bestFb.score = s0;
                        bestFb.angle = m_models[seed.modelIdx].angle;
                        bestFb.scale = m_models[seed.modelIdx].scale;
                        bestFb.scaleR = m_models[seed.modelIdx].scaleR;
                        bestFb.scaleC = m_models[seed.modelIdx].scaleC;
                        trackCoarsePromoted = true;
                    }
                }
            }

            // P8/P9 belt row-tracked L0 (Halcon §3.2.6): seeded Y rows only on borderline miss.
            const bool borderlineMiss = trackCoarsePromoted ? false
                : (bestRefined >= borderlineLo
                    || (!deduped.empty() && deduped[0].score >= userScore01 - 0.12f)
                    || (refined.empty() && !deduped.empty()
                        && deduped[0].score >= userScore01 - 0.06f));
            if (borderlineMiss && !ctxs.empty()) {
                if (!haveFbGrad) {
                    computeGradLevel(srcGray, fbNx, fbNy, fbMag);
                    haveFbGrad = true;
                }
                cv::Mat& rtNx = fbNx;
                cv::Mat& rtNy = fbNy;
                cv::Mat& rtMag = fbMag;
                const int stepNxR = (int)(rtNx.step / sizeof(float));
                const int stepMgR = (int)(rtMag.step / sizeof(float));
                const float* nxBaseR = rtNx.ptr<float>(0);
                const float* nyBaseR = rtNy.ptr<float>(0);
                const float* mgBaseR = rtMag.ptr<float>(0);
                int xMaxAllR = 0;
                for (const CoarseModelCtx& c : ctxs)
                    xMaxAllR = std::max(xMaxAllR, c.xMax);

                std::unordered_set<int> yRows;
                yRows.reserve(96);
                auto addYRow = [&](int y) {
                    if (y < 0 || y > globalYMax) return;
                    if (useHintYBand && hintPlY >= 0 && (y < yBandLo || y > yBandHi)) return;
                    yRows.insert(y);
                };
                if (useHintYBand && hintPlY >= 0) {
                    const int yStep = std::max(1, coarseScanStepYUsed);
                    for (int y = yBandLo; y <= yBandHi; y += yStep)
                        addYRow(y);
                }
                const size_t ySeedDedup = skipBeltDeepScan
                    ? std::min(deduped.size(), (size_t)12)
                    : std::min(deduped.size(), (size_t)32);
                for (size_t i = 0; i < ySeedDedup; ++i)
                    addYRow(deduped[i].y);
                const size_t ySeedCoarse = skipBeltDeepScan
                    ? std::min(topCands.size(), (size_t)16)
                    : std::min(topCands.size(), (size_t)64);
                for (size_t i = 0; i < ySeedCoarse; ++i)
                    addYRow(topCands[i].y);
                if (yRows.empty() && hintPlY >= 0)
                    addYRow(std::max(0, std::min(globalYMax, hintPlY)));

                const int refineR = std::max(coarseScanStepUsed + 12, 14);
                int maxW0R = 0, maxH0R = 0;
                for (int mi = 0; mi < numModels; ++mi) {
                    const ShapeModelLevel& ml0 = m_models[mi].levels[0];
                    maxW0R = std::max(maxW0R, ml0.width);
                    maxH0R = std::max(maxH0R, ml0.height);
                }

                auto rowTrackOneY = [&](int yRow) {
                    if (yRow > globalYMax) return;
                    int rowMi = -1, rowX = 0;
                    float rowS = -1.f;
                    for (int x = 0; x <= xMaxAllR; ++x) {
                        for (const CoarseModelCtx& c : ctxs) {
                            if (yRow > c.yMax || x > c.xMax) continue;
                            if (!maskAllowed(fineTopLevel, x, yRow, *c.ml)) continue;
                            float accPos = 0.f, accNeg = 0.f;
                            if (!coarseScoreAccumulate(
                                    x, yRow, c.n, c.feats, c.ox, c.oy,
                                    nxBaseR, nyBaseR, mgBaseR,
                                    stepNxR, stepMgR, fineMagFloor, metric,
                                    0.f, 0.f, accPos, accNeg))
                                continue;
                            float s = metricScore(accPos, accNeg, c.n, metric);
                            if (s > rowS) { rowS = s; rowMi = c.mi; rowX = x; }
                        }
                    }
                    if (rowMi < 0) return;
                    int bestX = rowX, bestY = yRow, bestMi = rowMi;
                    float bestS = rowS;
                    // P26 belt: alias X on a Y row — refine full X, not only around row peak (130530 @55_53).
                    const int xLoR = elongHorizTpl ? 0 : std::max(0, rowX - refineR);
                    const int yLoR = std::max(0, yRow - refineR);
                    const int xHiR = elongHorizTpl
                        ? (rtNx.cols - maxW0R) : std::min(rtNx.cols - maxW0R, rowX + refineR);
                    const int yHiR = std::min(rtNx.rows - maxH0R, yRow + refineR);
                    for (int yy = yLoR; yy <= yHiR; ++yy) {
                        for (int xx = xLoR; xx <= xHiR; ++xx) {
                            for (int mi = 0; mi < numModels; ++mi) {
                                const ShapeModelLevel& mlB = m_models[mi].levels[0];
                                if (mlB.features.empty()) continue;
                                if (xx > rtNx.cols - mlB.width || yy > rtNx.rows - mlB.height)
                                    continue;
                                float accPos = 0.f, accNeg = 0.f;
                                if (!coarseScoreAccumulate(
                                        xx, yy, (int)mlB.features.size(),
                                        mlB.features.data(), mlB.featOx.data(), mlB.featOy.data(),
                                        nxBaseR, nyBaseR, mgBaseR,
                                        stepNxR, stepMgR, fineMagFloor, metric,
                                        0.f, 0.f, accPos, accNeg))
                                    continue;
                                float s = metricScore(accPos, accNeg, (int)mlB.features.size(), metric);
                                if (s > bestS) {
                                    bestS = s;
                                    bestX = xx;
                                    bestY = yy;
                                    bestMi = mi;
                                }
                            }
                        }
                    }
                    const ShapeModelLevel& mlBest = m_models[bestMi].levels[0];
                    float sHal = scoreAtHalcon(rtNx, rtNy, rtMag,
                        (float)bestX, (float)bestY,
                        mlBest.features.data(), (int)mlBest.features.size(),
                        fineMagFloor, metric);
                    if (sHal > bestFb.score) {
                        bestFb.modelIdx = bestMi;
                        bestFb.x = (float)bestX;
                        bestFb.y = (float)bestY;
                        bestFb.score = sHal;
                        bestFb.angle = m_models[bestMi].angle;
                        bestFb.scale = m_models[bestMi].scale;
                        bestFb.scaleR = m_models[bestMi].scaleR;
                        bestFb.scaleC = m_models[bestMi].scaleC;
                    }
                };

                for (int yRow : yRows)
                    rowTrackOneY(yRow);

                // P9/P26: full row sweep on borderline (within 0.06 of MinScore).
                if (!skipBeltDeepScan
                    && bestFb.score + 1e-3f < userScore01
                    && bestFb.score >= userScore01 - 0.06f) {
                    const int yStep = std::max(1, coarseScanStepUsed);
                    for (int yRow = 0; yRow <= globalYMax; yRow += yStep) {
                        if (yRows.count(yRow)) continue;
                        rowTrackOneY(yRow);
                        if (bestFb.score + 1e-3f >= userScore01) break;
                    }
                }
            }

            if (!haveFbGrad)
                computeGradLevel(srcGray, fbNx, fbNy, fbMag);
            const int stepNxF = (int)(fbNx.step / sizeof(float));
            const int stepMgF = (int)(fbMag.step / sizeof(float));
            const float* nxBaseF = fbNx.ptr<float>(0);
            const float* nyBaseF = fbNy.ptr<float>(0);
            const float* mgBaseF = fbMag.ptr<float>(0);

            if (bestFb.score + 1e-3f < userScore01 && !skipBeltDeepScan) {
            for (size_t si = 0; si < seedK; ++si) {
                const RawCand& seed = deduped[si];
                const int cx0 = seed.x - fbOff.x;
                const int cy0 = seed.y - fbOff.y;
                const int xLoF = std::max(0, cx0 - expandRxF);
                const int yLoF = std::max(0, cy0 - expandRyF);
                for (int miF = 0; miF < numModels; ++miF) {
                    const ShapeModelLevel& mlF = m_models[miF].levels[0];
                    if (mlF.features.empty()) continue;
                    const int xHiF = std::min(fbNx.cols - mlF.width, cx0 + expandRxF);
                    const int yHiF = std::min(fbNx.rows - mlF.height, cy0 + expandRyF);
                    if (xHiF < xLoF || yHiF < yLoF) continue;
                    for (int yy = yLoF; yy <= yHiF; ++yy) {
                        for (int xx = xLoF; xx <= xHiF; ++xx) {
                            float accPos = 0.f, accNeg = 0.f;
                            if (!coarseScoreAccumulate(
                                    xx, yy, (int)mlF.features.size(),
                                    mlF.features.data(), mlF.featOx.data(), mlF.featOy.data(),
                                    nxBaseF, nyBaseF, mgBaseF,
                                    stepNxF, stepMgF, fineMagFloor, metric,
                                    0.f, 0.f, accPos, accNeg))
                                continue;
                            float s = metricScore(accPos, accNeg, (int)mlF.features.size(), metric);
                            if (s <= bestFb.score) continue;
                            const float sx = (float)xx + (float)fbOff.x;
                            const float sy = (float)yy + (float)fbOff.y;
                            float sHal = scoreAtHalcon(fbNx, fbNy, fbMag,
                                sx - (float)fbOff.x, sy - (float)fbOff.y,
                                mlF.features.data(), (int)mlF.features.size(),
                                fineMagFloor, metric);
                            if (sHal > bestFb.score) {
                                bestFb.modelIdx = miF;
                                bestFb.x = sx;
                                bestFb.y = sy;
                                bestFb.score = sHal;
                                bestFb.angle = m_models[miF].angle;
                                bestFb.scale = m_models[miF].scale;
                                bestFb.scaleR = m_models[miF].scaleR;
                                bestFb.scaleC = m_models[miF].scaleC;
                            }
                        }
                    }
                }
            }
            }
            if (bestFb.score + 1e-3f >= userScore01
                || (bestFb.score >= borderlineLo && m_subPixelMode != ShapeSubPixelMode::None)) {
                if (m_subPixelMode != ShapeSubPixelMode::None) {
                    const ShapeModelLevel& ml0fb = m_models[bestFb.modelIdx].levels[0];
                    cv::Mat spNx, spNy, spMag;
                    if (haveFbGrad) {
                        spNx = fbNx; spNy = fbNy; spMag = fbMag;
                    } else {
                        computeGradLevel(srcGray, spNx, spNy, spMag);
                    }
                    float v[3][3];
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx)
                            v[dy + 1][dx + 1] = scoreAtAlignment(spNx, spNy, spMag,
                                bestFb.x + (float)dx, bestFb.y + (float)dy,
                                ml0fb.features.data(), (int)ml0fb.features.size(),
                                fineMagFloor, metric);
                    double ox = 0., oy = 0.;
                    if (m_subPixelMode == ShapeSubPixelMode::LeastSquares)
                        quadraticSubpixelOffset2D(v, ox, oy);
                    else {
                        ox = parabolicPeakOffset(v[1][0], v[1][1], v[1][2]);
                        oy = parabolicPeakOffset(v[0][1], v[1][1], v[2][1]);
                    }
                    bestFb.x += (float)ox;
                    bestFb.y += (float)oy;
                    bestFb.score = scoreAtHalcon(spNx, spNy, spMag,
                        bestFb.x, bestFb.y,
                        ml0fb.features.data(), (int)ml0fb.features.size(),
                        fineMagFloor, metric);
                    // Steger/Halcon: sub-angle parabola on discrete angle models.
                    int mPrev = bestFb.modelIdx - 1, mNext = bestFb.modelIdx + 1;
                    if (mPrev < 0)          mPrev = angleWrap ? numModels - 1 : -1;
                    if (mNext >= numModels) mNext = angleWrap ? 0 : -1;
                    if (mPrev >= 0 && mNext >= 0) {
                        const ShapeModelLevel& mlP = m_models[mPrev].levels[0];
                        const ShapeModelLevel& mlN = m_models[mNext].levels[0];
                        if (!mlP.features.empty() && !mlN.features.empty()) {
                            float sP = scoreAtAlignment(spNx, spNy, spMag,
                                bestFb.x, bestFb.y,
                                mlP.features.data(), (int)mlP.features.size(),
                                fineMagFloor, metric);
                            float sN = scoreAtAlignment(spNx, spNy, spMag,
                                bestFb.x, bestFb.y,
                                mlN.features.data(), (int)mlN.features.size(),
                                fineMagFloor, metric);
                            float sC = bestFb.score;
                            double aoff = parabolicPeakOffset(sP, sC, sN);
                            bestFb.angle = (float)(m_models[bestFb.modelIdx].angle
                                + aoff * m_effectiveAngleStep);
                            bestFb.score = std::max(sP, std::max(sC, sN));
                        }
                    }
                }
                if (bestFb.score + 1e-3f >= userScore01)
                    refined.push_back(bestFb);
            }
            if (!refined.empty()) {
                std::sort(refined.begin(), refined.end(),
                    [](const Refined& a, const Refined& b) { return a.score > b.score; });
            }
        }
    }

    // P26 Halcon borderline: full-X L0 rescan (featStride=1) on seeded Y when still below MinScore.
    if (!skipL0Rescue && !skipBeltDeepScan && maxTargets == 1 && beltFindAtL0 && elongHorizTpl && largeScene
        && !deduped.empty() && beltCoarseAtL0 && !sp.nx[0].empty())
    {
        float bestSoFar = -1.f;
        for (const Refined& rf : refined)
            bestSoFar = std::max(bestSoFar, rf.score);
        const float coarseTop = deduped[0].score;
        const float probeBest = std::max(bestSoFar, coarseTop);
        // P27/P28: deep-scan only when refined empty and still borderline (Halcon §2.4.4).
        const bool deepNeeded = refined.empty()
            && bestSoFar + 1e-3f < userScore01
            && probeBest >= userScore01 - 0.06f;
        if (deepNeeded) {
            const cv::Mat& dnx = sp.nx[0];
            const cv::Mat& dny = sp.ny[0];
            const cv::Mat& dmag = sp.mag[0];
            const int stepNxD = (int)(dnx.step / sizeof(float));
            const int stepMgD = (int)(dmag.step / sizeof(float));
            const float* nxD = dnx.ptr<float>(0);
            const float* nyD = dny.ptr<float>(0);
            const float* mgD = dmag.ptr<float>(0);
            int xMaxD = 0, yMaxD = 0;
            for (int mi = 0; mi < numModels; ++mi) {
                const ShapeModelLevel& ml0 = m_models[mi].levels[0];
                if (ml0.features.empty()) continue;
                xMaxD = std::max(xMaxD, dnx.cols - ml0.width);
                yMaxD = std::max(yMaxD, dnx.rows - ml0.height);
            }
            std::unordered_set<int> yRowsD;
            yRowsD.reserve(64);
            for (size_t i = 0; i < std::min(deduped.size(), (size_t)32); ++i)
                yRowsD.insert(deduped[i].y);
            for (size_t i = 0; i < std::min(topCands.size(), (size_t)48); ++i)
                yRowsD.insert(topCands[i].y);
            Refined bestDeep;
            bestDeep.score = probeBest;
            for (int yRow : yRowsD) {
                if (yRow < 0 || yRow > yMaxD) continue;
                for (int x = 0; x <= xMaxD; x += 2) {
                    for (int mi = 0; mi < numModels; ++mi) {
                        const ShapeModelLevel& mlD = m_models[mi].levels[0];
                        if (mlD.features.empty()) continue;
                        if (x > dnx.cols - mlD.width || yRow > dnx.rows - mlD.height) continue;
                        if (!maskAllowed(0, x, yRow, mlD)) continue;
                        float accPos = 0.f, accNeg = 0.f;
                        if (!coarseScoreAccumulate(
                                x, yRow, (int)mlD.features.size(),
                                mlD.features.data(), mlD.featOx.data(), mlD.featOy.data(),
                                nxD, nyD, mgD, stepNxD, stepMgD, fineMagFloor, metric,
                                0.f, 0.f, accPos, accNeg))
                            continue;
                        float s = metricScore(accPos, accNeg, (int)mlD.features.size(), metric);
                        if (s <= bestDeep.score) continue;
                        float sHal = scoreAtHalcon(dnx, dny, dmag, (float)x, (float)yRow,
                            mlD.features.data(), (int)mlD.features.size(),
                            fineMagFloor, metric);
                        if (sHal > bestDeep.score) {
                            bestDeep.modelIdx = mi;
                            bestDeep.x = (float)x;
                            bestDeep.y = (float)yRow;
                            bestDeep.score = sHal;
                            bestDeep.angle = m_models[mi].angle;
                            bestDeep.scale = m_models[mi].scale;
                            bestDeep.scaleR = m_models[mi].scaleR;
                            bestDeep.scaleC = m_models[mi].scaleC;
                        }
                    }
                }
            }
            // P26/P27: full Y sweep only on tight borderline (Halcon robust-tracking fallback).
            if (bestDeep.score + 1e-3f < userScore01 && bestDeep.score >= userScore01 - 0.04f) {
                const int yStepD = std::max(2, coarseScanStepYUsed);
                for (int yRow = 0; yRow <= yMaxD; yRow += yStepD) {
                    if (yRowsD.count(yRow)) continue;
                    for (int x = 0; x <= xMaxD; x += 2) {
                        for (int mi = 0; mi < numModels; ++mi) {
                            const ShapeModelLevel& mlD = m_models[mi].levels[0];
                            if (mlD.features.empty()) continue;
                            if (x > dnx.cols - mlD.width || yRow > dnx.rows - mlD.height) continue;
                            if (!maskAllowed(0, x, yRow, mlD)) continue;
                            float accPos = 0.f, accNeg = 0.f;
                            if (!coarseScoreAccumulate(
                                    x, yRow, (int)mlD.features.size(),
                                    mlD.features.data(), mlD.featOx.data(), mlD.featOy.data(),
                                    nxD, nyD, mgD, stepNxD, stepMgD, fineMagFloor, metric,
                                    0.f, 0.f, accPos, accNeg))
                                continue;
                            float s = metricScore(accPos, accNeg, (int)mlD.features.size(), metric);
                            if (s <= bestDeep.score) continue;
                            float sHal = scoreAtHalcon(dnx, dny, dmag, (float)x, (float)yRow,
                                mlD.features.data(), (int)mlD.features.size(),
                                fineMagFloor, metric);
                            if (sHal > bestDeep.score) {
                                bestDeep.modelIdx = mi;
                                bestDeep.x = (float)x;
                                bestDeep.y = (float)yRow;
                                bestDeep.score = sHal;
                                bestDeep.angle = m_models[mi].angle;
                                bestDeep.scale = m_models[mi].scale;
                                bestDeep.scaleR = m_models[mi].scaleR;
                                bestDeep.scaleC = m_models[mi].scaleC;
                            }
                        }
                    }
                    if (bestDeep.score + 1e-3f >= userScore01) break;
                }
            }
            if (bestDeep.score + 1e-3f >= userScore01) {
                refined.clear();
                refined.push_back(bestDeep);
            }
        }
    }

    // P29/P36 Halcon pyramid_level_robust_tracking: L0 miss / mt=1 coarse-rank≠fine-rank.
    const bool needRobustTrack = refined.empty()
        || (maxTargets == 1 && !refined.empty() && refined[0].score + 1e-3f < userScore01);
    if (needRobustTrack && nLevels >= 2 && !deduped.empty()
        && deduped[0].score >= userScore01 - 0.12f)
    {
        const int robLevel = 1;
        ensureScenePyramid(srcGray, nLevels, robLevel, sp);
        const cv::Mat& rnx = sp.nx[robLevel];
        const cv::Mat& rny = sp.ny[robLevel];
        const cv::Mat& rmag = sp.mag[robLevel];
        if (!rnx.empty()) {
            const float magRob = coarseMagFloor(magFloor, nLevels, robLevel);
            const int localR = 6;
            const size_t tryK = std::min(deduped.size(), (size_t)8);
            Refined bestRob;
            bestRob.score = -1.f;
            for (size_t i = 0; i < tryK; ++i) {
                const RawCand& c = deduped[i];
                int cx = beltCoarseAtL0 ? (c.x / 2) : (c.x >> robLevel);
                int cy = beltCoarseAtL0 ? (c.y / 2) : (c.y >> robLevel);
                int bestMi = c.modelIdx;
                float bestS = -1.f;
                int bestX = cx, bestY = cy;
                const int xLo = std::max(0, cx - localR);
                const int yLo = std::max(0, cy - localR);
                for (int mi = 0; mi < numModels; ++mi) {
                    const ShapeModelLevel& mlR = m_models[mi].levels[robLevel];
                    if (mlR.features.empty()) continue;
                    const int xHi = std::min(rnx.cols - mlR.width, cx + localR);
                    const int yHi = std::min(rnx.rows - mlR.height, cy + localR);
                    if (xHi < xLo || yHi < yLo) continue;
                    for (int yy = yLo; yy <= yHi; ++yy) {
                        for (int xx = xLo; xx <= xHi; ++xx) {
                            if (!maskAllowed(robLevel, xx, yy, mlR)) continue;
                            float s = scoreAtHalcon(rnx, rny, rmag, (float)xx, (float)yy,
                                mlR.features.data(), (int)mlR.features.size(), magRob, metric);
                            if (s > bestS) {
                                bestS = s;
                                bestX = xx;
                                bestY = yy;
                                bestMi = mi;
                            }
                        }
                    }
                }
                if (bestS > bestRob.score) {
                    bestRob.modelIdx = bestMi;
                    bestRob.x = (float)bestX;
                    bestRob.y = (float)bestY;
                    bestRob.score = bestS;
                    bestRob.angle = m_models[bestMi].angle;
                    bestRob.scale = m_models[bestMi].scale;
                    bestRob.scaleR = m_models[bestMi].scaleR;
                    bestRob.scaleC = m_models[bestMi].scaleC;
                }
            }
            if (bestRob.score + 1e-3f >= acceptFine01) {
                const int up = beltCoarseAtL0 ? 2 : (1 << robLevel);
                bestRob.x *= (float)up;
                bestRob.y *= (float)up;
                if (refined.empty())
                    refined.push_back(bestRob);
                else
                    refined[0] = bestRob;
            }
        }
    }

    // P36/P40 NumMatches=1 non-belt: L0 local re-score top coarse seeds (154617 / 083933).
    if (maxTargets == 1 && !beltFindAtL0 && !elongHorizTpl
        && (refined.empty() || refined[0].score + 1e-3f < userScore01)
        && !deduped.empty() && !sp.nx[0].empty())
    {
        ensureScenePyramid(srcGray, nLevels, 0, sp);
        const cv::Mat& fnx = sp.nx[0];
        const cv::Mat& fny = sp.ny[0];
        const cv::Mat& fmag = sp.mag[0];
        const float mag0 = coarseMagFloor(magFloor, nLevels, 0);
        const int localR = largeTemplate ? 20 : 12;
        Refined bestL0;
        bestL0.score = refined.empty() ? -1.f : refined[0].score;
        const size_t tryK = std::min(deduped.size(),
            largeTemplate ? (size_t)24 : (size_t)12);
        for (size_t i = 0; i < tryK; ++i) {
            const RawCand& c = deduped[i];
            const int cx = c.x, cy = c.y;
            for (int mi = 0; mi < numModels; ++mi) {
                const ShapeModelLevel& ml0 = m_models[mi].levels[0];
                if (ml0.features.empty()) continue;
                const int xLo = std::max(0, cx - localR);
                const int yLo = std::max(0, cy - localR);
                const int xHi = std::min(fnx.cols - ml0.width, cx + localR);
                const int yHi = std::min(fnx.rows - ml0.height, cy + localR);
                for (int yy = yLo; yy <= yHi; ++yy) {
                    for (int xx = xLo; xx <= xHi; ++xx) {
                        if (!maskAllowed(0, xx, yy, ml0)) continue;
                        const float magLo = largeTemplate ? mag0 * 0.72f : mag0;
                        float s = scoreAtHalconBest(fnx, fny, fmag, (float)xx, (float)yy,
                            ml0.features.data(), (int)ml0.features.size(), magLo, metric,
                            largeTemplate);
                        if (s > bestL0.score) {
                            bestL0.modelIdx = mi;
                            bestL0.x = (float)xx;
                            bestL0.y = (float)yy;
                            bestL0.score = s;
                            bestL0.angle = m_models[mi].angle;
                            bestL0.scale = m_models[mi].scale;
                            bestL0.scaleR = m_models[mi].scaleR;
                            bestL0.scaleC = m_models[mi].scaleC;
                        }
                    }
                }
            }
        }
        if (bestL0.score + 1e-3f >= acceptFine01) {
            if (refined.empty())
                refined.push_back(bestL0);
            else
                refined[0] = bestL0;
        }
    }

    // P41 large template borderline: row-seeded L0 rescan (083933 ~0.49–0.51 vs Halcon ~0.7).
    if (maxTargets == 1 && largeTemplate && !elongHorizTpl && !skipL0Rescue
        && (refined.empty() || refined[0].score + 1e-3f < userScore01)
        && !sp.nx[0].empty())
    {
        float probeBest = refined.empty() ? -1.f : refined[0].score;
        if (!deduped.empty())
            probeBest = std::max(probeBest, deduped[0].score);
        if (!topCands.empty())
            probeBest = std::max(probeBest, topCands[0].score);
        if (probeBest >= userScore01 - 0.15f) {
            ensureScenePyramid(srcGray, nLevels, 0, sp);
            const cv::Mat& bnx = sp.nx[0];
            const cv::Mat& bny = sp.ny[0];
            const cv::Mat& bmag = sp.mag[0];
            const float magB = coarseMagFloor(magFloor, nLevels, 0);
            std::unordered_set<int> yRows;
            yRows.reserve(48);
            for (size_t i = 0; i < std::min(deduped.size(), (size_t)24); ++i)
                yRows.insert(deduped[i].y);
            for (size_t i = 0; i < std::min(topCands.size(), (size_t)32); ++i)
                yRows.insert(topCands[i].y);
            int xMaxB = 0, yMaxB = 0;
            for (int mi = 0; mi < numModels; ++mi) {
                const ShapeModelLevel& ml0 = m_models[mi].levels[0];
                if (ml0.features.empty()) continue;
                xMaxB = std::max(xMaxB, bnx.cols - ml0.width);
                yMaxB = std::max(yMaxB, bnx.rows - ml0.height);
            }
            Refined bestRow;
            bestRow.score = probeBest;
            for (int yRow : yRows) {
                if (yRow < 0 || yRow > yMaxB) continue;
                const int xStep = (probeBest >= userScore01 - 0.06f) ? 1 : 2;
                for (int x = 0; x <= xMaxB; x += xStep) {
                    for (int mi = 0; mi < numModels; ++mi) {
                        const ShapeModelLevel& ml0 = m_models[mi].levels[0];
                        if (ml0.features.empty()) continue;
                        if (!maskAllowed(0, x, yRow, ml0)) continue;
                        const float magLo = largeTemplate ? magB * 0.72f : magB;
                        float s = scoreAtHalconBest(bnx, bny, bmag, (float)x, (float)yRow,
                            ml0.features.data(), (int)ml0.features.size(), magLo, metric,
                            largeTemplate);
                        if (s > bestRow.score) {
                            bestRow.modelIdx = mi;
                            bestRow.x = (float)x;
                            bestRow.y = (float)yRow;
                            bestRow.score = s;
                            bestRow.angle = m_models[mi].angle;
                            bestRow.scale = m_models[mi].scale;
                            bestRow.scaleR = m_models[mi].scaleR;
                            bestRow.scaleC = m_models[mi].scaleC;
                        }
                    }
                }
            }
            if (bestRow.score + 1e-3f >= acceptFine01) {
                if (refined.empty())
                    refined.push_back(bestRow);
                else
                    refined[0] = bestRow;
            }
        }
    }

    // P42b: coarse-cluster L0 rescue for non-belt ambiguous scenes (circle multi-station).
    if (maxTargets == 1 && !elongHorizTpl && !beltFindAtL0 && !skipL0Rescue
        && (refined.empty() || refined[0].score + 1e-3f < userScore01)
        && deduped.size() >= 2 && !sp.nx[0].empty())
    {
        float cx = 0.f, cy = 0.f, wSum = 0.f;
        const size_t ck = std::min(deduped.size(), (size_t)8);
        for (size_t i = 0; i < ck; ++i) {
            cx += deduped[i].x * deduped[i].score;
            cy += deduped[i].y * deduped[i].score;
            wSum += deduped[i].score;
        }
        if (wSum > 1e-4f) {
            cx /= wSum;
            cy /= wSum;
            ensureScenePyramid(srcGray, nLevels, 0, sp);
            const cv::Mat& cnx = sp.nx[0];
            const cv::Mat& cny = sp.ny[0];
            const cv::Mat& cmag = sp.mag[0];
            const float magC = coarseMagFloor(magFloor, nLevels, 0);
            const int localR = largeTemplate ? 28 : 18;
            Refined bestCl;
            bestCl.score = refined.empty() ? -1.f : refined[0].score;
            const int xLo = std::max(0, (int)std::lround(cx) - localR);
            const int yLo = std::max(0, (int)std::lround(cy) - localR);
            for (int mi = 0; mi < numModels; ++mi) {
                const ShapeModelLevel& ml0 = m_models[mi].levels[0];
                if (ml0.features.empty()) continue;
                const int xHi = std::min(cnx.cols - ml0.width, (int)std::lround(cx) + localR);
                const int yHi = std::min(cnx.rows - ml0.height, (int)std::lround(cy) + localR);
                for (int yy = yLo; yy <= yHi; ++yy) {
                    for (int xx = xLo; xx <= xHi; ++xx) {
                        if (!maskAllowed(0, xx, yy, ml0)) continue;
                        float s = scoreAtHalconBest(cnx, cny, cmag, (float)xx, (float)yy,
                            ml0.features.data(), (int)ml0.features.size(), magC, metric,
                            largeTemplate);
                        if (s > bestCl.score) {
                            bestCl.modelIdx = mi;
                            bestCl.x = (float)xx;
                            bestCl.y = (float)yy;
                            bestCl.score = s;
                            bestCl.angle = m_models[mi].angle;
                            bestCl.scale = m_models[mi].scale;
                            bestCl.scaleR = m_models[mi].scaleR;
                            bestCl.scaleC = m_models[mi].scaleC;
                        }
                    }
                }
            }
            if (bestCl.score + 1e-3f >= acceptFine01) {
                if (refined.empty())
                    refined.push_back(bestCl);
                else
                    refined[0] = bestCl;
            }
        }
    }

    // P42a/P57: large-template — always L0 visible rescore (083933 score≥ gate was
    // never reached when coarse score already >= MinScore); ICP only when borderline miss.
    if (maxTargets == 1 && largeTemplate && !refined.empty() && !sp.nx[0].empty()) {
        Refined& r = refined[0];
        ensureScenePyramid(srcGray, nLevels, 0, sp);
        const cv::Mat& fnx = sp.nx[0];
        const cv::Mat& fny = sp.ny[0];
        const cv::Mat& fmag = sp.mag[0];
        const float mag0 = coarseMagFloor(magFloor, nLevels, 0) * 0.72f;
        const ShapeModelLevel& ml0 = m_models[r.modelIdx].levels[0];
        if (!ml0.features.empty()) {
            float ang = r.angle;
            icpRefinePose(fnx, fny, fmag, r.x, r.y, ang, ml0, mag0, metric);
            r.angle = ang;
            const float magFull = coarseMagFloor(magFloor, nLevels, 0);
            r.score = std::max(
                scoreAtHalconBest(fnx, fny, fmag, r.x, r.y,
                    ml0.features.data(), (int)ml0.features.size(), mag0, metric, true),
                scoreAtHalconBest(fnx, fny, fmag, r.x, r.y,
                    ml0.features.data(), (int)ml0.features.size(), magFull, metric, true));
        }
    }

    // P42e/P44: compact template — L0 visible rescore; Y-bucket disambiguation (circle multi-station).
    if (maxTargets == 1 && compactTemplate && !deduped.empty() && !sp.nx[0].empty())
    {
        const bool multiStripe = deduped.size() >= 2
            && (std::abs(deduped[0].x - deduped[1].x) > std::max(tplRefW, 80)
                || std::abs(deduped[0].y - deduped[1].y) > std::max(tplRefH * 4, 80));
        const bool needCompactRescore = refined.empty()
            || refined[0].score + 1e-3f < userScore01
            || (multiStripe && metric != ShapeMatchMetric::IgnoreLocalPolarity);
        if (needCompactRescore) {
        ensureScenePyramid(srcGray, nLevels, 0, sp);
        const cv::Mat& fnx = sp.nx[0];
        const cv::Mat& fny = sp.ny[0];
        const cv::Mat& fmag = sp.mag[0];
        const float mag0 = coarseMagFloor(magFloor, nLevels, 0);
        const int localR = 14;
        Refined bestCmp;
        bestCmp.score = refined.empty() ? -1.f : refined[0].score;
        const size_t tryK = std::min(deduped.size(), multiStripe ? (size_t)4 : (size_t)8);
        int bestYBucket = -1;
        if (multiStripe && metric != ShapeMatchMetric::IgnoreLocalPolarity) {
            const int yBucketH = std::max(tplRefH * 2, 40);
            std::unordered_map<int, float> yBucketMax;
            for (size_t i = 0; i < tryK; ++i) {
                const int by = deduped[i].y / yBucketH;
                yBucketMax[by] = std::max(yBucketMax[by], deduped[i].score);
            }
            float bestW = -1.f;
            for (const auto& kv : yBucketMax) {
                if (kv.second > bestW + 1e-4f
                    || (std::fabs(kv.second - bestW) <= 1e-4f && kv.first > bestYBucket)) {
                    bestW = kv.second;
                    bestYBucket = kv.first;
                }
            }
            // P44 circle: when buckets are close, prefer higher-Y station (true part vs top alias).
            int altBucket = -1;
            float altW = -1.f;
            for (const auto& kv : yBucketMax) {
                if (kv.first == bestYBucket) continue;
                if (kv.second > altW) {
                    altW = kv.second;
                    altBucket = kv.first;
                }
            }
            if (altBucket >= 0 && altW >= bestW * 0.55f && altBucket > bestYBucket)
                bestYBucket = altBucket;
        }
        // P44: compact rescore — when current hit is low-Y alias, prefer higher-Y bucket if competitive.
        if (multiStripe && !refined.empty() && metric != ShapeMatchMetric::IgnoreLocalPolarity) {
            const int yBucketH = std::max(tplRefH * 2, 40);
            const int hitBucket = (int)std::floor(refined[0].y / (float)yBucketH);
            if (bestYBucket >= 0 && bestYBucket > hitBucket)
                bestCmp.score = -1.f;
        }
        for (size_t i = 0; i < tryK; ++i) {
            const RawCand& c = deduped[i];
            if (bestYBucket >= 0) {
                const int yBucketH = std::max(tplRefH * 2, 40);
                if (c.y / yBucketH != bestYBucket) continue;
            }
            const int cx = c.x, cy = c.y;
            for (int mi = 0; mi < numModels; ++mi) {
                const ShapeModelLevel& ml0 = m_models[mi].levels[0];
                if (ml0.features.empty()) continue;
                const int xLo = std::max(0, cx - localR);
                const int yLo = std::max(0, cy - localR);
                const int xHi = std::min(fnx.cols - ml0.width, cx + localR);
                const int yHi = std::min(fnx.rows - ml0.height, cy + localR);
                for (int yy = yLo; yy <= yHi; ++yy) {
                    for (int xx = xLo; xx <= xHi; ++xx) {
                        if (!maskAllowed(0, xx, yy, ml0)) continue;
                        float px = (float)xx, py = (float)yy;
                        float ang = m_models[mi].angle;
                        float s = scoreAtHalconVisible(fnx, fny, fmag, px, py,
                            ml0.features.data(), (int)ml0.features.size(), mag0, metric);
                        if (s + 1e-3f < acceptFine01 - 0.05f) continue;
                        icpRefinePose(fnx, fny, fmag, px, py, ang, ml0, mag0, metric);
                        s = scoreAtHalconBest(fnx, fny, fmag, px, py,
                            ml0.features.data(), (int)ml0.features.size(), mag0, metric, true);
                        if (s > bestCmp.score) {
                            bestCmp.modelIdx = mi;
                            bestCmp.x = px;
                            bestCmp.y = py;
                            bestCmp.score = s;
                            bestCmp.angle = ang;
                            bestCmp.scale = m_models[mi].scale;
                            bestCmp.scaleR = m_models[mi].scaleR;
                            bestCmp.scaleC = m_models[mi].scaleC;
                        }
                    }
                }
            }
        }
        if (bestCmp.score + 1e-3f >= acceptFine01) {
            if (refined.empty())
                refined.push_back(bestCmp);
            else
                refined[0] = bestCmp;
        }
        }
    }

    // P42b: ambiguous top peaks — prefer pose near score-weighted coarse cluster.
    if (maxTargets == 1 && refined.size() >= 2 && !elongHorizTpl && deduped.size() >= 2) {
        float cx = 0.f, wSum = 0.f;
        const size_t ck = std::min(deduped.size(), (size_t)8);
        for (size_t i = 0; i < ck; ++i) {
            cx += deduped[i].x * deduped[i].score;
            wSum += deduped[i].score;
        }
        if (wSum > 1e-4f) {
            cx /= wSum;
            std::sort(refined.begin(), refined.end(),
                [](const Refined& a, const Refined& b) { return a.score > b.score; });
            const float topS = refined[0].score;
            const float scoreEps = 0.10f;
            size_t bestIdx = 0;
            float bestDist = std::numeric_limits<float>::max();
            for (size_t i = 0; i < refined.size(); ++i) {
                if (refined[i].score + scoreEps < topS) break;
                const float dist = std::abs(refined[i].x - cx);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = i;
                }
            }
            if (bestIdx != 0)
                std::swap(refined[0], refined[bestIdx]);
        }
    }

    // P32 belt: among near-top refined poses, prefer closest to motion/median X prior.
    const float coarseTopGapFinal = topCands.size() >= 2
        ? (topCands[0].score - topCands[1].score) : 1.f;
    if (maxTargets == 1 && beltFindAtL0 && elongHorizTpl && refined.size() >= 2
        && coarseTopGapFinal < 0.10f) {
        float priorPlX = beltMedianPlX;
        if (m_hasFindHint && m_trackLastFind && !m_models.empty()) {
            const ShapeModelLevel& mlRef = m_models[0].levels[0];
            priorPlX = (float)(m_findHintCol - searchOffset.x - mlRef.originX
                - m_positionBiasX);
        }
        if (priorPlX >= 0.f) {
            std::sort(refined.begin(), refined.end(),
                [](const Refined& a, const Refined& b) { return a.score > b.score; });
            const float topS = refined[0].score;
            const float scoreEps = 0.04f;
            size_t bestIdx = 0;
            float bestDist = std::numeric_limits<float>::max();
            for (size_t i = 0; i < refined.size(); ++i) {
                if (refined[i].score + scoreEps < topS) break;
                const float dist = std::abs(refined[i].x - priorPlX);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = i;
                }
            }
            if (bestIdx != 0)
                std::swap(refined[0], refined[bestIdx]);
        }
    }

    // P43 belt track miss: wrong-stripe hint can exhaust fast path with no hit — full-X L0 rescan.
    if (maxTargets == 1 && refined.empty() && beltFindAtL0 && elongHorizTpl && largeScene
        && m_trackLastFind && m_hasFindHint && !sp.nx[0].empty() && !deduped.empty())
    {
        ensureScenePyramid(srcGray, nLevels, 0, sp);
        const cv::Mat& rnx = sp.nx[0];
        const cv::Mat& rny = sp.ny[0];
        const cv::Mat& rmag = sp.mag[0];
        const float mag0 = coarseMagFloor(magFloor, nLevels, 0);
        int xMaxR = 0, yMaxR = 0;
        for (int mi = 0; mi < numModels; ++mi) {
            const ShapeModelLevel& ml0 = m_models[mi].levels[0];
            if (ml0.features.empty()) continue;
            xMaxR = std::max(xMaxR, rnx.cols - ml0.width);
            yMaxR = std::max(yMaxR, rnx.rows - ml0.height);
        }
        std::unordered_set<int> yRows;
        yRows.reserve(16);
        for (size_t i = 0; i < std::min(deduped.size(), (size_t)8); ++i)
            yRows.insert(deduped[i].y);
        for (size_t i = 0; i < std::min(topCands.size(), (size_t)8); ++i)
            yRows.insert(topCands[i].y);
        Refined bestRec;
        bestRec.score = -1.f;
        for (int yRow : yRows) {
            if (yRow < 0 || yRow > yMaxR) continue;
            for (int x = 0; x <= xMaxR; x += 2) {
                for (int mi = 0; mi < numModels; ++mi) {
                    const ShapeModelLevel& ml0 = m_models[mi].levels[0];
                    if (ml0.features.empty()) continue;
                    if (!maskAllowed(0, x, yRow, ml0)) continue;
                    float s = scoreAtHalconBest(rnx, rny, rmag, (float)x, (float)yRow,
                        ml0.features.data(), (int)ml0.features.size(), mag0, metric, false);
                    if (s > bestRec.score) {
                        bestRec.modelIdx = mi;
                        bestRec.x = (float)x;
                        bestRec.y = (float)yRow;
                        bestRec.score = s;
                        bestRec.angle = m_models[mi].angle;
                        bestRec.scale = m_models[mi].scale;
                        bestRec.scaleR = m_models[mi].scaleR;
                        bestRec.scaleC = m_models[mi].scaleC;
                    }
                }
            }
        }
        if (bestRec.score + 1e-3f >= userScore01)
            refined.push_back(bestRec);
    }

    if (refined.empty()) return emptyOut;

    // ===== STAGE 3: convert to MatchInfo + rotated-rect NMS. =====
    //
    // Result position is the centre of the original (unpadded) template, computed
    // from the per-model origin within the cropped bounding box.
    std::vector<MatchInfo> out;
    out.reserve(refined.size());
    for (const auto& r : refined) {
        const ShapeModelLevel& ml0 = m_models[r.modelIdx].levels[0];
        cv::Point2d center(
            r.x + ml0.originX + searchOffset.x + m_positionBiasX,
            r.y + ml0.originY + searchOffset.y + m_positionBiasY);
        out.emplace_back(
            std::max(0.0, std::min(1.0, (double)r.score)),
            (double)r.angle,
            center,
            (double)r.scale,
            m_models[r.modelIdx].templateId,
            (double)r.scaleR,
            (double)r.scaleC);
    }

    std::sort(out.begin(), out.end(),
              [](const MatchInfo& a, const MatchInfo& b) { return a.score > b.score; });

    auto refSize = [&](int modelId, float scR, float scC) -> cv::Size2f {
        if (modelId >= 0 && modelId < (int)m_templateMeta.size()) {
            const auto& tm = m_templateMeta[modelId];
            return cv::Size2f((float)tm.refW * scC, (float)tm.refH * scR);
        }
        return cv::Size2f((float)m_templateRefW * scC, (float)m_templateRefH * scR);
    };

    std::vector<char> killed(out.size(), 0);
    for (size_t i = 0; i < out.size(); ++i) {
        if (killed[i]) continue;
        const float scR = (float)out[i].scaleR;
        const float scC = (float)out[i].scaleC;
        cv::RotatedRect ri(
            cv::Point2f((float)out[i].position.x, (float)out[i].position.y),
            refSize(out[i].modelId, scR, scC),
            (float)out[i].angle);
        for (size_t j = i + 1; j < out.size(); ++j) {
            if (killed[j]) continue;
            const float scR2 = (float)out[j].scaleR;
            const float scC2 = (float)out[j].scaleC;
            cv::RotatedRect rj(
                cv::Point2f((float)out[j].position.x, (float)out[j].position.y),
                refSize(out[j].modelId, scR2, scC2),
                (float)out[j].angle);
            if (rotatedRectIoU(ri, rj) > m_maxOverlap) killed[j] = 1;
        }
    }

    std::vector<MatchInfo> finalOut;
    finalOut.reserve(maxTargets);
    for (size_t i = 0; i < out.size(); ++i) {
        if (!killed[i]) {
            finalOut.push_back(out[i]);
            if ((int)finalOut.size() >= maxTargets) break;
        }
    }

    if (m_trackLastFind && !finalOut.empty()
        && finalOut[0].score + 1e-3f >= userScore01) {
        m_findHintCol = finalOut[0].position.x;
        m_findHintRow = finalOut[0].position.y;
        m_findHintScore = finalOut[0].score;
        m_hasFindHint = true;
    }

    return finalOut;
}

} // namespace pattern_matching
} // namespace cv
