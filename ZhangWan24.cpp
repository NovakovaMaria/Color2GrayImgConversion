/*
* Author of the code: Mária Nováková
* Author of the paper: Zhang L., Wan Y.
* Date of implementation: 28.04.2024
* Description: Implementation of color-to-gray image conversion using salient colors and radial basis functions.
*/

#include "ZhangWan24.hpp"

// CIELab of a BGR image, normalised to [0,1] like the paper normalises all values to [0,1]:
// L/100, (a+128)/255, (b+128)/255 (this is OpenCV's 8-bit Lab encoding divided by 255)
static Mat normalizedLab(const Mat &bgr8) {
    Mat bgr, lab;
    bgr8.convertTo(bgr, CV_32F, 1/255.0);
    cvtColor(bgr, lab, COLOR_BGR2Lab);                       // L in [0,100], a and b around [-128,127]
    lab = lab.reshape(1, static_cast<int>(lab.total()));
    lab.col(0) *= 1/100.0;
    lab.col(1) = (lab.col(1) + 128.0) / 255.0;
    lab.col(2) = (lab.col(2) + 128.0) / 255.0;
    return lab.reshape(3, bgr8.rows);
}

// normalised Lab (see normalizedLab) of one colour given as BGR in [0,1]
static Vec3f labOfBGR(const Vec3f &bgr) {
    Mat px(1, 1, CV_32FC3, Scalar(bgr[0], bgr[1], bgr[2])), lab;
    cvtColor(px, lab, COLOR_BGR2Lab);
    Vec3f t = lab.at<Vec3f>(0, 0);
    return Vec3f(t[0] / 100.0f, (t[1] + 128.0f) / 255.0f, (t[2] + 128.0f) / 255.0f);
}

// RGB values (BGR order, [0,1]) of the pixels of one cluster as an N x 1 three-channel matrix,
// used for the split of Eq. (1), which is done in RGB (paper, Fig. 4 and Fig. 5)
static Mat clusterPixels(const Mat &rgbImage, const vector<pair<Point, Vec3f>> &cluster) {
    Mat data(static_cast<int>(cluster.size()), 1, CV_32FC3);
    for (size_t i = 0; i < cluster.size(); i++) data.at<Vec3f>(static_cast<int>(i)) = rgbImage.at<Vec3f>(cluster[i].first);
    return data;
}

/************ PART 1 ************/

/**
 * @brief First step of the conversion, determine salient colors 
 * 
 * @param image input image
 * @param k current number of salient colors
 * @param max_k maximum number for quantizied colors
 * @param theta_0 threshold value for MSE for synthetic images
 * @param theta_1 threshold value for Mk for natural images
 */
void ColorToGrayConverter::quantizeColors(Mat &image, int &k, int max_k, float theta_0, float theta_1) {
    
    cv::Mat imagefloat, imageLab, grayImage, grayImage8, output1;

    // image to grayscale
    image.convertTo(imagefloat, CV_32F, 1/255.0);
    cvtColor(imagefloat, grayImage, COLOR_BGR2GRAY);
    cvtColor(image, grayImage8, COLOR_BGR2GRAY); // 8-bit gray for the entropy histogram

    // image to CIE colors space, normalised to [0,1]
    imageLab = normalizedLab(image);

    // colours are quantised in RGB with the perceptual distance in CIELab (paper, Fig. 4b and 5b "our method"):
    // centroids are mean RGB colours, stored as their Lab values so that all distances are CIELab distances (Eq. 2)
    this->rgbImage = imagefloat;

    // determine type of the image (synthetic vs natural)
    float E = Entropy(grayImage8);
    bool classif = classification(E);

    // determine first centroid color, it is mean of all values in the image
    Scalar meanScalar = mean(imagefloat);
    float mse_k_prev, m_k_prev;

    Vec3f c_0 = labOfBGR(Vec3f(static_cast<float>(meanScalar[0]), 
                    static_cast<float>(meanScalar[1]), 
                    static_cast<float>(meanScalar[2])));

    vector<Vec3f> centers;
    centers.push_back(c_0);

    // create initial clusters
    vector<vector<pair<Point, Vec3f>>> clusters = clusterImage(imageLab, centers);

    float prevMSE_k = numeric_limits<float>::max();
    float mse_k;

    // adjust clusters and recompute centers
    do {
        mse_k = MSE_k(imageLab, centers, clusters);

        // STEP 5
        if (mse_k == 0 || ((abs(mse_k - prevMSE_k)) / prevMSE_k) < pow(10, -6)) break;

        clusters = clusterImage(imageLab, centers);

        actualizeCenters(&centers, clusters);

        prevMSE_k = mse_k;

    } while (true); // STEP 6

    float mseg_k = MSEG_k(grayImage, centers, clusters);

    float m_k = M_k(mse_k, mseg_k);

    mse_k_prev = mse_k;
    m_k_prev = m_k;

    vector<float> individualMSE;

    // save the best colors
    vector<Vec3f> best_centers;
    vector<vector<pair<Point, Vec3f>>> best_clusters;

    int position_mse;
    
    // create new centroids from selected centroid (in this case c_0)
    // (take the stored centre: after the updates above it is no longer bit-identical to c_0,
    //  and expandCentroids() finds the centre to replace by exact comparison)
    c_0 = centers[0];
    expandCentroids(c_0, k, clusterPixels(rgbImage, clusters[0]), &centers);

    // iterate till maximum number condition is not met
    while (k <= max_k) {
        // STEP 2
        clusters = clusterImage(imageLab, centers);

        // STEP 3
        actualizeCenters(&centers, clusters);

        // STEP 4
        prevMSE_k = numeric_limits<float>::max();
        do {

            mse_k = MSE_k(imageLab, centers, clusters);

            // STEP 5
            if (mse_k == 0 || ((abs(mse_k - prevMSE_k)) / prevMSE_k) < pow(10, -6)) break;

            clusters = clusterImage(imageLab, centers);

            actualizeCenters(&centers, clusters);

            prevMSE_k = mse_k;

        } while (true); // STEP 6

        // STEP 7
        best_centers = centers;
        best_clusters = clusters;

        mseg_k = MSEG_k(grayImage, centers, clusters);

        m_k = M_k(mse_k, mseg_k);

        // select centroid for new expansion, it is the one with biggest MSE
        // (MSE of the converged clusters of this iteration only)
        individualMSE.clear();
        MSE(centers, clusters, &individualMSE);

        float minMSE = -1.0f;
        position_mse = 0;

        for (size_t i = 0; i < individualMSE.size(); i++){
            if (minMSE < individualMSE[i]){
                minMSE = individualMSE[i];
                position_mse = i;
            }
        }

        c_0 = centers[position_mse];

        // principal direction of the pixels that belong to c_0 (paper, Sec. 3.1)
        expandCentroids(c_0, k, clusterPixels(rgbImage, clusters[position_mse]), &centers);

        clusters = clusterImage(imageLab, centers);

        if(classif){ // synthetic
            if(mse_k <= theta_0 && mse_k_prev > theta_0){
                break;
            }
        }else{ // natural
            if(m_k <= theta_1 && m_k_prev > theta_1){
                break;
            }
        }

        m_k_prev = m_k;
        mse_k_prev = mse_k;
    }

    // save quantizied image
    auto output = convertToQuantizedImage(image,best_centers,best_clusters);

    cvtColor(output, output1, COLOR_Lab2BGR);
    imwrite("quantizied_step1.png", output1);
    
    this->centers = best_centers;
    this->clusters = best_clusters;
}

/************ PART 2 ************/

/**
 * @brief Ordering grayscale colors
 * 
 * @param image input image
 * @param method 1 = order by rgb2gray value of the quantized colors (paper Sec. 3.2.1, Eq. 11)
 *               2 = order by weighted distance from the basic color (paper Sec. 3.2.2, Algorithm 2)
 */
void ColorToGrayConverter::ordering(Mat image, int method){

    vector<vector<pair<Point, Vec3f>>> clusters = this->clusters;
    vector<Vec3f> centers = this->centers;

    float min_distance = -1.0f;
    float distance;
    int i0, i1 = 0, i2 = 0, k = centers.size();

    vector<float> grey(k); 

    // select two colors with the biggest distance (needed by method 2 only)
    for (int i = 0; i < k - 1; i++){
        Vec3f color1 = centers[i];
        for (int j = i+1; j < k; j++){
            Vec3f color2 = centers[j];
            distance = weightedEuclidean(color1, color2);

            if (min_distance < distance){
                i1 = i;
                i2 = j;
                min_distance = distance;
            }
        }
    }

    // select basic color from two colors based on lower L component
    i0 = centers[i1][0] < centers[i2][0] ? i1 : i2;

    vector<pair<int, float>> storage;

    Vec3f basic_color = centers[i0];

    // method 1 (Eq. 11): sort key is the rgb2gray value of the quantizied color
    // method 2 (Eq. 13): sort key is the distance between basic color and the quantizied color
    for (int i = 0; i < k; i++){
        Vec3f color = centers[i];
        distance = (method == 1) ? rgb2grayOfCenter(color) : weightedEuclidean(color, basic_color);
        storage.push_back(make_pair(i,distance));
    }

    // colors colors by distance
    sort(storage.begin(), storage.end(), 
        [](const pair<int, float>& a, const pair<int, float>& b) {
            if (a.second != b.second) {
                return a.second < b.second; 
            }
            return a.first < b.first;
        }
    );

    // assign evenly spaced grey colors to quantizied colors based on the sorted keys (Eq. 11 / Eq. 14)
    for (int m = 1; m <= k; m++){
        int index = storage[m-1].first;
        grey[index] = static_cast<float>(m - 1) / (k - 1);
    }

    this->grayvalues = grey;

    // store assigned grey colors in image
    Mat output1;
    auto output = convertToGrayQuantizedImage(image,grey,clusters);
    cvtColor(output, output1, COLOR_RGB2GRAY);
    imwrite("gray_withoutstep3.png", output1);
    
}

/************ PART 3 ************/

/**
 * @brief Create grayscale image based on ordered gray values and radial basis function
 * 
 * @param image input image
 * @param sigma scaling parameter for Laplace Kernel
 */
void ColorToGrayConverter::createGrayScale(Mat image, float sigma, bool edgeFix){
    vector<Vec3f> centers = this->centers;
    int k = centers.size();

    vector<float> gray = this->grayvalues; 
    vector<float> a(k);

    // determine weights a_j by solving the k x k linear system of Eq. (16):
    //   g_i = sum_j a_j * phi(x_i, x_j),  i = 1..k   <=>   Phi * a = g
    Mat Phi(k, k, CV_64F), G(k, 1, CV_64F), A;
    for (int i = 0; i < k; i++){
        G.at<double>(i) = gray[i];
        for (int j = 0; j < k; j++){
            Phi.at<double>(i, j) = laplaceKernel(centers[i], centers[j], sigma);
        }
    }
    if (!solve(Phi, G, A, DECOMP_LU)) solve(Phi, G, A, DECOMP_SVD);
    for (int i = 0; i < k; i++) a[i] = static_cast<float>(A.at<double>(i));

    float f_x;

    Mat output = image.clone(), imageLab, grayImage;

    imageLab = normalizedLab(image);

    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // assign gray value to each pixel (Eq. 16-18)
    Mat F(image.rows, image.cols, CV_32F);
    for (int x = 0; x < image.rows; x++){
        for (int y = 0; y < image.cols; y++){
            Vec3f img_color = imageLab.at<Vec3f>(x,y);

            f_x = getGreyValue(img_color, a, sigma);

            F.at<float>(x, y) = clamp(f_x);
        }
    }

    // optional extension, NOT part of the paper
    if (edgeFix) F = repairEdgePixels(image, F);

    for (int x = 0; x < image.rows; x++){
        for (int y = 0; y < image.cols; y++){
            grayImage.at<uchar>(x, y) = static_cast<uchar>(F.at<float>(x, y)*255);
        }
    }
    
    // store result
    imwrite("result_bw.png", grayImage);
    
}

/************ PART 1 - HELPER FUNCTIONS ************/

/**
 * @brief Compute PCA direction from the image
 * 
 * @param image input image
 * @return Vec3f direction vector
 */
Vec3f ColorToGrayConverter::computePrincipalDirection(const Mat& image) {
    Mat data = image.reshape(1, image.total()); // reshape to a single row per pixel
    data.convertTo(data, CV_32F);

    PCA pca(data, Mat(), PCA::DATA_AS_ROW, 1); // keep only the first principal component
    Vec3f D_PCA;
    for (int i = 0; i < 3; ++i) {
        D_PCA[i] = pca.eigenvectors.at<float>(0, i);
    }
    return D_PCA;   
}

/**
 * @brief Expanding centroids with the given formula, new centroids are computed from selected one with the highest MSE
 * 
 * @param c_0 coordinates of centroid from which two new are computed
 * @param k current number of quantizied colors
 * @param img RGB pixels that belong to c_0, used for the split colour and the PCA direction
 * @param centers centroids (quantizied colors)
 */
void ColorToGrayConverter::expandCentroids(Vec3f c_0, int &k, Mat img, vector<Vec3f> *centers) {
    // delta = 1/255, the minimum increment of an 8-bit value normalised to [0,1] (paper, Eq. 1)
    const float delta = 1.0f / 255.0f;

    // compute PCA (in RGB, as the split is done in RGB)
    Vec3f D_pca = computePrincipalDirection(img);

    // RGB colour of c_0 = mean RGB colour of its pixels
    Scalar m = mean(img);
    Vec3f c_0_rgb(static_cast<float>(m[0]), static_cast<float>(m[1]), static_cast<float>(m[2]));

    // compute new centroids / colors (Eq. 1 in RGB), stored as Lab like all centroids
    Vec3f N_c1 = labOfBGR(c_0_rgb + delta * D_pca);
    Vec3f N_c2 = labOfBGR(c_0_rgb - delta * D_pca);

    // remove old color
    auto it = find(centers->begin(), centers->end(), c_0);
    if (it != centers->end()) {
        centers->erase(it);
    }

    // add two new colors
    centers->push_back(N_c1);
    centers->push_back(N_c2);

    // increment current quantizied colors
    k++;
}

/**
 * @brief Compute Euclidean Distance
 * 
 * @param color1 color value
 * @param color2 color value (most of the time quantizied color)
 * @return float distance
 */
float ColorToGrayConverter::euclideanDistance(const Vec3f& color1, const Vec3f& color2) {
    float dL = color1[0] - color2[0];
    float da = color1[1] - color2[1];
    float db = color1[2] - color2[2];
    return sqrt(dL * dL + da * da + db * db);
}

/**
 * @brief Create clusters
 * 
 * @param image input image
 * @param centroids centroid colors
 * @return vector<vector<pair<Point, Vec3f>>> clusters of input image
 */
vector<vector<pair<Point, Vec3f>>> ColorToGrayConverter::clusterImage(Mat image, vector<Vec3f> centroids) {
    vector<vector<pair<Point, Vec3f>>> clusters(centroids.size());

    for (int x = 0; x < image.rows; x++) {
        for (int y = 0; y < image.cols; y++) {
            Vec3f pixel = image.at<Vec3f>(x,y);
            float minDistance = numeric_limits<float>::max();
            int assignedCluster = 0;

            for (size_t i = 0; i < centroids.size(); i++) {
                float distance = (euclideanDistance(pixel, centroids[i]));

                if (distance < minDistance) {
                    minDistance = distance;
                    assignedCluster = i;
                }
            }

            clusters[assignedCluster].push_back(make_pair(Point(y,x),pixel));
        }
    }

    return clusters;
}

/**
 * @brief Update centers based on current colors of pixels assigned to corresponding cluster
 *        (new center = mean RGB colour of the cluster, stored as its Lab value)
 * 
 * @param centers centroids of the clusters
 * @param clusters clusters of image
 */
void ColorToGrayConverter::actualizeCenters(vector<Vec3f> *centers, vector<vector<pair<Point, Vec3f>>> clusters) {
    if (centers->empty()) {
        cerr << "Centers vector is empty." << endl;
        return;
    }

    for (size_t i = 0; i < clusters.size(); ++i) {
        Vec3d meanVal(0.0, 0.0, 0.0);
        size_t clusterSize = clusters[i].size();

        if (clusterSize == 0) {
            cerr << "Cluster " << i << " is empty. Skipping mean calculation." << endl;
            continue;
        }

        for (const auto& pixel : clusters[i]) {
            meanVal += Vec3d(rgbImage.at<Vec3f>(pixel.first));
        }

        Vec3f newCenter = labOfBGR(Vec3f(meanVal / static_cast<double>(clusterSize)));

        (*centers)[i] = newCenter; 
    }
}

/**
 * @brief Compute MSE for individual clusters (for synthetic and natural)
 * 
 * @param centers centroids of the clusters
 * @param clusters clusters of image
 * @param individualMSE vector which holds MSE value for each cluster
 */
void ColorToGrayConverter::MSE(vector<Vec3f> centers, vector<vector<pair<Point, Vec3f>>> clusters, vector<float> *individualMSE) {
    
    for (size_t i = 0; i < centers.size(); i++) {
        float tmp = 0;
        Vec3f color_cluster = centers[i];
        for (size_t j = 0; j < clusters[i].size(); j++) {
            Vec3f color = clusters[i][j].second;
            float diff = euclideanDistance(color, color_cluster);
            tmp += diff*diff;
        }
        if (clusters[i].size() > 0) {
            individualMSE->push_back(tmp / clusters[i].size());
        } else {
            individualMSE->push_back(0);
        }
    }
}

/**
 * @brief Compute MSE for whole image (for synthetic and natural)
 * 
 * @param image input image
 * @param centers centroids of the clusters
 * @param clusters clusters of image
 * @return float MSE value
 */
float ColorToGrayConverter::MSE_k(Mat image, vector<Vec3f> centers, vector<vector<pair<Point, Vec3f>>> clusters) {
    float mse = 0.0;

    for (size_t i = 0; i < centers.size(); i++) {
        Vec3f color_cluster = centers[i];

        for (const auto& cluster_pixel : clusters[i]) {
            Point coords = cluster_pixel.first;
            Vec3f color = image.at<Vec3f>(coords);
            float diff = euclideanDistance(color, color_cluster);
            mse += diff*diff;
        }
    }

    return mse / static_cast<float>(image.total());
}

/**
 * @brief Compute average gray color of the image's clusters
 * 
 * @param image grayscale image
 * @param cluster clusters of image
 * @return float average color of cluster
 */
float ColorToGrayConverter::g(Mat image, vector<pair<Point, Vec3f>> cluster){
    float Qx = 0.0;
    for (const auto& cluster_pixel : cluster) {
            Point coords = cluster_pixel.first;
            float color = image.at<float>(coords);
            Qx += color;
        }
    return Qx / cluster.size();
}

/**
 * @brief Compute MSE for grayscaled image
 * 
 * @param image grayscale image
 * @param centers centroids of the clusters
 * @param clusters clusters of image
 * @return float MSE value
 */
float ColorToGrayConverter::MSEG_k(Mat image, vector<Vec3f> centers, vector<vector<pair<Point, Vec3f>>> clusters){
    float mse = 0.0;

    for (size_t i = 0; i < centers.size(); i++) {

        float gQx = g(image, clusters[i]);

        for (const auto& cluster_pixel : clusters[i]) {
            Point coords = cluster_pixel.first;
            float color = image.at<float>(coords);

            // squared difference between color in the grayscale image and computed average graycolor in the cluster
            // (Eq. 5 prints |.|, but MSEG is a mean SQUARE error and Fig. 7 shows it on the scale of MSE)
            float diff = color - gQx;
            mse += diff * diff;
        }
    }

    return mse / static_cast<float>(image.total());
}

/**
 * @brief Compute Entropy value, used for determinings type of the image
 * 
 * @param img input image
 * @return float entropy value
 */
float ColorToGrayConverter::Entropy(Mat img){

    vector<int> histogram(256, 0);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            histogram[img.at<uchar>(i, j)]++;
        }
    }

    int total_pixels = img.rows * img.cols;

    vector<float> p(256);

    // compute probabilities of gray color in the image
    for (int i = 0; i < 256; i++) {
        p[i] = (float)histogram[i] / total_pixels;
    }

    float E = 0;
    for (size_t i = 0; i < p.size(); i++){
        if (p[i] > 0) E += p[i]*log2(p[i]); // bits (max 8 for 8-bit image), 0*log(0) is taken as 0
    }
    
    return -E;
}

/**
 * @brief Determine type of the image
 * 
 * @param E Entropy value
 * @return true return value for synthetic image
 * @return false return value for natural image
 */
bool ColorToGrayConverter::classification(float E){
    return(E <= 4 ? true : false); // true for synthetic, false for natural
}

/**
 * @brief Compute combined MSE score (for natural)
 * 
 * @param MSE_k MSE of quantizied colors
 * @param MSEG_k MSE of grayscale image
 * @return float MSE value
 */
float ColorToGrayConverter::M_k(float MSE_k, float MSEG_k) {
    return sqrt(MSE_k * MSEG_k);
}

/************ PART 2 - HELPER FUNCTIONS ************/

/**
 * @brief rgb2gray value of a quantizied color (Eq. 11), in [0,1]
 * 
 * @param labColor quantizied color as stored in centers (normalised Lab: L/100, (a+128)/255, (b+128)/255)
 * @return float gray value
 */
float ColorToGrayConverter::rgb2grayOfCenter(const Vec3f& labColor){
    // convert back to RGB through true Lab values (L in [0,100], a and b centered on 0)
    Mat lab(1, 1, CV_32FC3, Scalar(labColor[0] * 100.0f, labColor[1] * 255.0f - 128.0f, labColor[2] * 255.0f - 128.0f)), bgr;
    cvtColor(lab, bgr, COLOR_Lab2BGR);
    Vec3f c = bgr.at<Vec3f>(0, 0);
    return 0.2989f * c[2] + 0.5870f * c[1] + 0.1140f * c[0]; // weights of the paper
}

/**
 * @brief Compute weighted Euclidean Distance
 * 
 * @param color1 color value 
 * @param color2 color value
 * @return float distance
 */
float ColorToGrayConverter::weightedEuclidean(const Vec3f& color1, const Vec3f& color2){
    float dL = color1[0] - color2[0];
    float da = color1[1] - color2[1];
    float db = color1[2] - color2[2];
    return sqrt(dL * dL * 0.6 + da * da * 0.3 + db * db * 0.1);
}

/************ PART 3 - HELPER FUNCTIONS ************/

/**
 * @brief EXTENSION, NOT PART OF THE PAPER: repair the gray value of blended (anti-aliased) edge pixels.
 *
 * The method maps colours to gray values without looking at the position of a pixel. A pixel on the
 * border of two regions often has a blended colour c_p = (1-t)*c_A + t*c_B of the colours on both sides,
 * and such a colour can be mapped to a gray value outside the range of the two sides, which shows up
 * as thin dark or white lines along region borders.
 *
 * For every pixel, pairs of pixels on opposite sides (distance 1 and 2, horizontal, vertical and both
 * diagonals) are tested. If the pixel's RGB colour lies on the segment between the two colours
 * (residual <= 0.04, 0 <= t <= 1) and the two colours really differ (distance >= 0.1), the pair with
 * the largest colour difference is used. If the pixel's gray value lies outside the gray range of that
 * pair (by more than 4/255), or differs from the same blend of the two gray values, (1-t)*g_A + t*g_B,
 * by more than 8/255, it is replaced by that blend. Pixels with 4 or more neighbours of nearly the same
 * colour (difference < 0.02) belong to a flat region and are skipped: a region colour can happen to lie
 * between two other colours, which would otherwise be mistaken for a blend where several regions meet.
 * All other pixels are left unchanged. (Correcting
 * only out-of-range pixels left the in-range ones of the same border unchanged, which gave stair-stepped
 * borders.)
 *
 * @param image input image (BGR, 8-bit)
 * @param F gray values in [0,1] from Eq. (18)
 * @return Mat repaired gray values in [0,1]
 */
Mat ColorToGrayConverter::repairEdgePixels(const Mat &image, const Mat &F){
    const float maxResidual = 0.04f, minEdge = 0.1f, eps = 4.0f / 255.0f, maxBlendError = 8.0f / 255.0f;
    const float sameColour = 0.01f, newRegionColour = 0.02f;
    const int dirs[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    const int R = image.rows, C = image.cols;
    auto inside = [&](int x, int y){ return x >= 0 && y >= 0 && x < R && y < C; };

    Mat rgb;
    image.convertTo(rgb, CV_32F, 1/255.0);
    Mat out = F.clone();
    int changed = 0;

    // flat = pixel with 4 or more near-identical neighbours (inside a region, not on a thin blended border)
    Mat flat = Mat::zeros(R, C, CV_8U);
    for (int x = 0; x < R; x++){
        for (int y = 0; y < C; y++){
            Vec3f c = rgb.at<Vec3f>(x, y);
            int same = 0;
            for (int dx = -1; dx <= 1; dx++){
                for (int dy = -1; dy <= 1; dy++){
                    if ((dx == 0 && dy == 0) || !inside(x + dx, y + dy)) continue;
                    Vec3f dn = rgb.at<Vec3f>(x + dx, y + dy) - c;
                    if (dn.dot(dn) < sameColour * sameColour) same++;
                }
            }
            if (same >= 4) flat.at<uchar>(x, y) = 1;
        }
    }
    // member = flat, or the same colour as a flat neighbour. The gray is a function of the colour,
    // so such a pixel already has the gray of its region and must not be changed.
    Mat member = flat.clone();
    for (int x = 0; x < R; x++){
        for (int y = 0; y < C; y++){
            if (member.at<uchar>(x, y)) continue;
            Vec3f c = rgb.at<Vec3f>(x, y);
            for (int dx = -1; dx <= 1; dx++){
                for (int dy = -1; dy <= 1; dy++){
                    if (!inside(x + dx, y + dy) || !flat.at<uchar>(x + dx, y + dy)) continue;
                    Vec3f dn = rgb.at<Vec3f>(x + dx, y + dy) - c;
                    if (dn.dot(dn) < sameColour * sameColour) member.at<uchar>(x, y) = 1;
                }
            }
        }
    }

    // ---- 1) thin border between two regions: blend of the two colours on opposite sides ----
    for (int x = 0; x < R; x++){
        for (int y = 0; y < C; y++){
            if (member.at<uchar>(x, y)) continue;
            Vec3f c = rgb.at<Vec3f>(x, y);
            float bestEdge = -1.0f, bestT = 0.0f, gA = 0.0f, gB = 0.0f;

            for (const auto &d : dirs){
                for (int s = 1; s <= 2; s++){
                    int xa = x - s * d[0], ya = y - s * d[1], xb = x + s * d[0], yb = y + s * d[1];
                    if (!inside(xa, ya) || !inside(xb, yb)) continue;

                    Vec3f cA = rgb.at<Vec3f>(xa, ya), cB = rgb.at<Vec3f>(xb, yb), v = cB - cA;
                    float len2 = v.dot(v);
                    if (len2 < minEdge * minEdge) continue;              // no real edge between A and B

                    float t = (c - cA).dot(v) / len2;
                    if (t < 0.0f || t > 1.0f) continue;                  // not between the two colours
                    Vec3f r = c - (cA + t * v);
                    if (r.dot(r) > maxResidual * maxResidual) continue;  // not a blend of the two colours

                    if (len2 > bestEdge){
                        bestEdge = len2; bestT = t;
                        gA = F.at<float>(xa, ya); gB = F.at<float>(xb, yb);
                    }
                }
            }

            if (bestEdge < 0.0f) continue;
            float g = F.at<float>(x, y);
            float blend = (1.0f - bestT) * gA + bestT * gB;
            if (g < min(gA, gB) - eps || g > max(gA, gB) + eps || fabs(g - blend) > maxBlendError){
                out.at<float>(x, y) = blend;
                changed++;
            }
        }
    }

    // ---- 2) junctions: where 3 or more regions meet, a pixel blends up to 3 colours. Fit its colour as a
    // convex combination of 1-3 colours of the flat regions within radius 2 and use the same weights on
    // their grays. Only these region grays are used, so a wrong border pixel cannot spread.
    int junction = 0;
    for (int x = 0; x < R; x++){
        for (int y = 0; y < C; y++){
            if (member.at<uchar>(x, y)) continue;
            vector<Vec3f> col; vector<float> gr;
            for (int dx = -2; dx <= 2; dx++){
                for (int dy = -2; dy <= 2; dy++){
                    if (!inside(x + dx, y + dy) || !flat.at<uchar>(x + dx, y + dy)) continue;
                    Vec3f cn = rgb.at<Vec3f>(x + dx, y + dy);
                    bool known = false;
                    for (const auto &k : col){ Vec3f dn = cn - k; if (dn.dot(dn) < newRegionColour * newRegionColour) known = true; }
                    if (!known){ col.push_back(cn); gr.push_back(F.at<float>(x + dx, y + dy)); }
                }
            }
            if (col.empty()) continue;

            Vec3f c = rgb.at<Vec3f>(x, y);
            const int n = col.size();
            float bestRes = 1e9f, pred = 0.0f;
            auto take = [&](float res, float p){ if (res < bestRes - 1e-4f){ bestRes = res; pred = p; } };
            for (int i = 0; i < n; i++)                                   // one region colour
                take(norm(c - col[i]), gr[i]);
            for (int i = 0; i < n; i++){                                  // blend of two
                for (int j = i + 1; j < n; j++){
                    Vec3f v = col[j] - col[i]; float l = v.dot(v);
                    if (l < 1e-8f) continue;
                    float t = (c - col[i]).dot(v) / l;
                    if (t < 0.0f || t > 1.0f) continue;
                    take(norm(c - col[i] - t * v), (1.0f - t) * gr[i] + t * gr[j]);
                }
            }
            for (int i = 0; i < n; i++){                                  // blend of three
                for (int j = i + 1; j < n; j++){
                    for (int k = j + 1; k < n; k++){
                        Vec3f u = col[j] - col[i], v = col[k] - col[i], q = c - col[i];
                        float uu = u.dot(u), uv = u.dot(v), vv = v.dot(v), det = uu * vv - uv * uv;
                        if (fabs(det) < 1e-10f) continue;
                        float a = (vv * u.dot(q) - uv * v.dot(q)) / det, b = (uu * v.dot(q) - uv * u.dot(q)) / det;
                        if (a < 0.0f || b < 0.0f || a + b > 1.0f) continue;
                        take(norm(q - a * u - b * v), (1.0f - a - b) * gr[i] + a * gr[j] + b * gr[k]);
                    }
                }
            }
            if (bestRes > maxResidual) continue;                          // colour is not a blend of the regions
            if (fabs(out.at<float>(x, y) - pred) > maxBlendError){
                if (out.at<float>(x, y) == F.at<float>(x, y)) changed++;
                out.at<float>(x, y) = pred;
                junction++;
            }
        }
    }

    cerr << "edge repair (extension): " << changed << " pixels changed ("
         << 100.0 * changed / image.total() << " %), " << junction << " of them at junctions" << endl;
    return out;
}

/**
 * @brief RBF function - Gaussian
 * 
 * @param color1 color value
 * @param color2 color value
 * @param sigma standard deviation
 * @return float 
 */
float ColorToGrayConverter::gaussianKernel(const Vec3f& color1, const Vec3f& color2, float sigma){
    float euclid = euclideanDistance(color1, color2);
    return exp(- euclid * euclid / (2*sigma*sigma)); // Eq. (15) is printed without the square
}

/**
 * @brief RBF function - Laplace
 * 
 * @param color1 color value
 * @param color2 color value
 * @param sigma standard deviation
 * @return float 
 */
float ColorToGrayConverter::laplaceKernel(const Vec3f& color1, const Vec3f& color2, float sigma){
    float euclid = euclideanDistance(color1, color2);
    return exp(- euclid / sigma);
}

/**
 * @brief Clamp color to [0,1]
 * 
 * @param value color value
 * @return float normalized value
 */
float ColorToGrayConverter::clamp(float value) {
    if (value < 0.0f) return 0.0f;
    if (value > 1.0f) return 1.0f;
    return value;
}

/**
 * @brief Compute grayvalue with usage of RBF (this implementation uses laplace)
 * 
 * @param img_color color value
 * @param a weights
 * @param sigma standard deviation
 * @return float gray value
 */
float ColorToGrayConverter::getGreyValue(Vec3f img_color, vector<float> a, float sigma){
    vector<Vec3f> centers = this->centers;

    float f_x = 0.0;
    for (size_t i = 0; i < centers.size(); i++){
        Vec3f color1 = centers[i]; 
        f_x += a[i] * laplaceKernel(img_color, color1, sigma);
    }
    return f_x;
}


/************ FUNCTIONS FOR CONVERSION, FOR THE IMAGE OUTPUT ************/


/**
 * @brief Convert image to gray image
 * 
 * @param originalImage input image
 * @param centroids assigned gray color to quantizied colors
 * @param clusters clusters of image
 * @return Mat gray image
 */
Mat ColorToGrayConverter::convertToGrayQuantizedImage(const Mat& originalImage, const vector<float>& centroids, const vector<vector<pair<Point, Vec3f>>>& clusters) {
    Mat quantizedImage = originalImage.clone();
    for (size_t clusterIndex = 0; clusterIndex < clusters.size(); ++clusterIndex) {
        for (const auto& pixel : clusters[clusterIndex]) {
            uchar v = saturate_cast<uchar>(centroids[clusterIndex] * 255);
            quantizedImage.at<Vec3b>(pixel.first) = Vec3b(v, v, v);
        }
    }

    return quantizedImage;
}

/**
 * @brief Convert image to quantizied image
 * 
 * @param originalImage input image
 * @param centroids quantizied colors
 * @param clusters clusters of image
 * @return Mat quantizied image
 */
Mat ColorToGrayConverter::convertToQuantizedImage(const Mat& originalImage, const vector<Vec3f>& centroids, const vector<vector<pair<Point, Vec3f>>>& clusters) {
    Mat quantizedImage = originalImage.clone();
    for (size_t clusterIndex = 0; clusterIndex < clusters.size(); ++clusterIndex) {
        // normalised Lab -> OpenCV 8-bit Lab encoding (converted to BGR by the caller)
        Vec3b centroidColor = Vec3b(
            saturate_cast<uchar>(centroids[clusterIndex][0] * 255.0f),
            saturate_cast<uchar>(centroids[clusterIndex][1] * 255.0f),
            saturate_cast<uchar>(centroids[clusterIndex][2] * 255.0f)
        );

        for (const auto& pixel : clusters[clusterIndex]) {
            quantizedImage.at<Vec3b>(pixel.first) = centroidColor;
        }
    }

    return quantizedImage;
}

/**
 * @brief Main function
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char* argv[]) {
    
    if (argc < 4 || argc > 6) {
        cerr << "Usage: " << argv[0] << " <image_path> <max_k> <sensitivity> [<ordering>] [<edge_repair>]\n";
        cerr << "Please provide an image path, max number of clusters (max_k), and sensitivity.\n";
        cerr << "Optional ordering: 1 = rgb2gray ordering (Eq. 11), 2 = distance ordering (Algorithm 2, default).\n";
        cerr << "Optional edge_repair: 1 = repair blended edge pixels (extension, not in the paper), 0 = off (default).\n";
        return 1;
    }

    Mat image = imread(argv[1]);
    if (image.empty()) {
        cerr << "Could not open or find the image: " << argv[1] << "\n";
        return 1;
    }

    ColorToGrayConverter converter;

    float sigma = 100; 

    if(argc >= 3){
        sigma = atof(argv[3]);
    }

    int k = 1;
    int max_k = atoi(argv[2]);
    float theta_0 = 0.0004;  // paper value (values normalised to [0,1])
    float theta_1 = 0.00065; // paper value (values normalised to [0,1])
    
    converter.quantizeColors(image, k, max_k, theta_0, theta_1);
    converter.ordering(image, (argc >= 5) ? atoi(argv[4]) : 2);
    converter.createGrayScale(image, sigma, (argc == 6) && atoi(argv[5]) == 1);
    
    return 0;
}