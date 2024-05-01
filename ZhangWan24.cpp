#include "ZhangWan24.hpp"


void ColorToGrayConverter::quantizeColors(Mat &image, int &k, int max_k, float theta_0, float theta_1) {
                    
    cv::Mat imagefloat, imageLab, imageBGR, grayImage, output1;
    image.convertTo(imagefloat, CV_32F, 1/255.0);
    
    cvtColor(image, imageLab, COLOR_BGR2Lab);
    
    imageLab.convertTo(imageLab, CV_32F);

    cvtColor(imagefloat, grayImage, COLOR_BGR2GRAY);

    Scalar meanScalar = mean(imageLab);
    float mse_k_prev, m_k_prev;

    Vec3f c_0(static_cast<float>(meanScalar[0]), 
                    static_cast<float>(meanScalar[1]), 
                    static_cast<float>(meanScalar[2]));

    vector<Vec3f> centers;
    centers.push_back(c_0);

    vector<vector<pair<Point, Vec3f>>> clusters = clusterImage(imageLab, centers);

    float prevMSE_k = numeric_limits<float>::max();
    float mse_k;

    do {
        mse_k = MSE_k(imageLab, centers, clusters);

        // STEP 5
        if (((abs(mse_k - prevMSE_k)) / prevMSE_k) < pow(10, -6)) break;

        clusters = clusterImage(imageLab, centers);

        actualizeCenters(&centers, clusters);

        prevMSE_k = mse_k;

    } while (true); // STEP 6

    float mseg_k = MSEG_k(grayImage, centers, clusters);

    float m_k = M_k(mse_k, mseg_k);

    mse_k_prev = mse_k;
    m_k_prev = m_k;

    vector<float> individualMSE;

    vector<Vec3f> best_centers;
    vector<vector<pair<Point, Vec3f>>> best_clusters;

    int position_mse;
    
    expandCentroids(c_0, k, imageLab, &centers);

    float E = Entropy(grayImage);

    bool classif = classification(E);

    while (k <= max_k) {
        // STEP 2
        clusters = clusterImage(imageLab, centers);

        // STEP 3
        actualizeCenters(&centers, clusters);

        MSE(centers, clusters, &individualMSE);

        // STEP 4
        prevMSE_k = numeric_limits<float>::max();
        do {

            mse_k = MSE_k(imageLab, centers, clusters);

            // STEP 5
            if (((abs(mse_k - prevMSE_k)) / prevMSE_k) < pow(10, -6)) break;

            clusters = clusterImage(imageLab, centers);

            actualizeCenters(&centers, clusters);

            prevMSE_k = mse_k;

        } while (true); // STEP 6

        // STEP 7
        best_centers = centers;
        best_clusters = clusters;

        mseg_k = MSEG_k(grayImage, centers, clusters);

        m_k = M_k(mse_k, mseg_k);

        float minMSE = numeric_limits<float>::min();

        for (size_t i = 0; i < individualMSE.size(); i++){
            if (minMSE < individualMSE[i]){
                minMSE = individualMSE[i];
                position_mse = i;
            }
        }

        c_0 = centers[position_mse];
        expandCentroids(c_0, k, imageLab, &centers);

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

    auto output = convertToQuantizedImage(image,best_centers,best_clusters);

    cvtColor(output, output1, COLOR_Lab2BGR);
    imwrite("quantizied_step1.png", output1);
    
    this->centers = best_centers;
    this->clusters = best_clusters;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ColorToGrayConverter::ordering(Mat image){

    vector<vector<pair<Point, Vec3f>>> clusters = this->clusters;
    vector<Vec3f> centers = this->centers;

    float min_distance = numeric_limits<float>::min();
    float distance;
    int i0, i1, i2, k = centers.size();

    vector<float> grey(k); 

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

    i0 = centers[i1][0] < centers[i2][0] ? i1 : i2;

    vector<pair<int, float>> storage;

    Vec3f basic_color = centers[i0];

    for (int i = 0; i < k; i++){
        Vec3f color = centers[i];
        distance = weightedEuclidean(color, basic_color);
        storage.push_back(make_pair(i,distance));
    }

    sort(storage.begin(), storage.end(), 
        [](const pair<int, float>& a, const pair<int, float>& b) {
            if (a.second != b.second) {
                return a.second < b.second; 
            }
            return a.first < b.first;
        }
    );

    for (int m = 1; m <= k; m++){
        int index = storage[m-1].first;
        grey[index] = static_cast<float>(m - 1) / (k - 1);
    }

    this->grayvalues = grey;

    Mat output1;
    auto output = convertToGrayQuantizedImage(image,grey,clusters);
    cvtColor(output, output1, COLOR_RGB2GRAY);
    imwrite("gray_withoutstep3.png", output1);
    
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void ColorToGrayConverter::createGrayScale(Mat image, float sigma){
    vector<Vec3f> centers = this->centers;
    int k = centers.size();

    vector<float> gray = this->grayvalues; 
    vector<float> a(k); 

    float sum;

    for (int i = 0; i < k; i++){ 
        Vec3f color1 = centers[i]; 
        float greycolor1 = gray[i];
        sum = 0.0;
        for (int j = 0; j < k; j++){
                Vec3f color2 = centers[j]; 
                sum += laplaceKernel(color1, color2, sigma);             
        }
        a[i] = greycolor1 / sum;
    }

    float f_x;

    Mat output = image.clone(), imageLab, imagefloat, grayImage;

    cvtColor(image, imageLab, COLOR_BGR2Lab);
    imageLab.convertTo(imageLab, CV_32F);

    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    for (int x = 0; x < image.rows; x++){
        for (int y = 0; y < image.cols; y++){
            Vec3f img_color = imageLab.at<Vec3f>(x,y);

            f_x = getGreyValue(img_color, a, sigma);

            grayImage.at<uchar>(x, y) = static_cast<uchar>(clamp(f_x)*255);
        }
    }

    

    imwrite("gray_withstep3.png", grayImage);
    
}

//////////////////////// part 1  ////////////////////////

Vec3f ColorToGrayConverter::computePrincipalDirection(const Mat& image) {
    Mat data = image.reshape(1, image.total()); // Reshape to a single row per pixel
    data.convertTo(data, CV_32F);

    PCA pca(data, Mat(), PCA::DATA_AS_ROW, 1); // Keep only the first principal component
    Vec3f D_PCA;
    for (int i = 0; i < 3; ++i) {
        D_PCA[i] = pca.eigenvectors.at<float>(0, i);
    }
    return D_PCA;   
}

void ColorToGrayConverter::expandCentroids(Vec3f c_0, int &k, Mat img, vector<Vec3f> *centers) {
    Vec3f sigma(1.0f / 255.0f,1.0f / 255.0f, 1.0f / 255.0f);

    Vec3f D_pca = computePrincipalDirection(img);

    Vec3f N_c1 = c_0 + sigma * D_pca;
    Vec3f N_c2 = c_0 - sigma * D_pca;

    auto it = find(centers->begin(), centers->end(), c_0);
    if (it != centers->end()) {
        centers->erase(it);
    }

    centers->push_back(N_c1);
    centers->push_back(N_c2);
    k++;
}

float ColorToGrayConverter::euclideanDistance(const Vec3f& color1, const Vec3f& color2) {
    float dL = color1[0] - color2[0];
    float da = color1[1] - color2[1];
    float db = color1[2] - color2[2];
    return sqrt(dL * dL + da * da + db * db);
}

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

void ColorToGrayConverter::actualizeCenters(vector<Vec3f> *centers, vector<vector<pair<Point, Vec3f>>> clusters) {
    if (centers->empty()) {
        cerr << "Centers vector is empty." << endl;
        return;
    }

    for (size_t i = 0; i < clusters.size(); ++i) {
        Vec3f meanVal(0.0, 0.0, 0.0);
        size_t clusterSize = clusters[i].size();

        if (clusterSize == 0) {
            cerr << "Cluster " << i << " is empty. Skipping mean calculation." << endl;
            continue;
        }

        for (const auto& pixel : clusters[i]) {
            meanVal += pixel.second;
        }

        Vec3f newCenter = meanVal / static_cast<float>(clusterSize);

        (*centers)[i] = newCenter; 
    }
}

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

float ColorToGrayConverter::g(Mat image, vector<pair<Point, Vec3f>> cluster){
    float Qx = 0.0;
    for (const auto& cluster_pixel : cluster) {
            Point coords = cluster_pixel.first;
            float color = image.at<float>(coords);
            Qx += color;

        }

    return Qx / cluster.size();
}

float ColorToGrayConverter::MSEG_k(Mat image, vector<Vec3f> centers, vector<vector<pair<Point, Vec3f>>> clusters){
    float mse = 0.0;

    for (size_t i = 0; i < centers.size(); i++) {

        float gQx = g(image, clusters[i]);

        for (const auto& cluster_pixel : clusters[i]) {
            Point coords = cluster_pixel.first;
            float color = image.at<float>(coords);

            float diff = abs(color - gQx);
            mse += diff;
        }
    }

    return mse / static_cast<float>(image.total());
}


float ColorToGrayConverter::Entropy(Mat img){

    vector<int> histogram(256, 0);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            histogram[img.at<uchar>(i, j)]++;
        }
    }

    int total_pixels = img.rows * img.cols;

    vector<float> p(256);

    for (int i = 0; i < 256; i++) {
        p[i] = (float)histogram[i] / total_pixels;
    }

    float E = 0;
    for (size_t i = 0; i < p.size(); i++){
        E += p[i]*log(p[i]);
    }
    
    return -E;
}

bool ColorToGrayConverter::classification(float E){
    return(E <= 4 ? true : false); // true for synthetic, false for natural
}

float ColorToGrayConverter::M_k(float MSE_k, float MSEG_k) {
    return sqrt(MSE_k * MSEG_k);
}

//////////////////////// part 2  ////////////////////////

float ColorToGrayConverter::weightedEuclidean(const Vec3f& color1, const Vec3f& color2){
    float dL = color1[0] - color2[0];
    float da = color1[1] - color2[1];
    float db = color1[2] - color2[2];
    return sqrt(dL * dL * 0.6 + da * da * 0.3 + db * db * 0.1);
}

//////////////////////// part 3  ////////////////////////

float ColorToGrayConverter::gaussianKernel(const Vec3f& color1, const Vec3f& color2, float sigma){
    float euclid = euclideanDistance(color1, color2);
    return exp(- euclid / (2*sigma*sigma));
}

float ColorToGrayConverter::laplaceKernel(const Vec3f& color1, const Vec3f& color2, float sigma){
    float euclid = euclideanDistance(color1, color2);
    return exp(- euclid / sigma);
}

float ColorToGrayConverter::clamp(float value) {
    if (value < 0.0f) return 0.0f;
    if (value > 1.0f) return 1.0f;
    return value;
}

float ColorToGrayConverter::getGreyValue(Vec3f img_color, vector<float> a, float sigma){
    vector<Vec3f> centers = this->centers;

    float f_x = 0.0;
    for (size_t i = 0; i < centers.size(); i++){
        Vec3f color1 = centers[i]; 
        f_x += a[i] * laplaceKernel(img_color, color1, sigma);
    }
    return f_x;
}


// image plots
Mat ColorToGrayConverter::convertToGrayQuantizedImage(const Mat& originalImage, const vector<float>& centroids, const vector<vector<pair<Point, Vec3f>>>& clusters) {
    Mat quantizedImage = originalImage.clone();
    for (size_t clusterIndex = 0; clusterIndex < clusters.size(); ++clusterIndex) {
        for (const auto& pixel : clusters[clusterIndex]) {
            quantizedImage.at<Vec3b>(pixel.first) = centroids[clusterIndex] * 255;
        }
    }

    return quantizedImage;
}

Mat ColorToGrayConverter::convertToQuantizedImage(const Mat& originalImage, const vector<Vec3f>& centroids, const vector<vector<pair<Point, Vec3f>>>& clusters) {
    Mat quantizedImage = originalImage.clone();
    for (size_t clusterIndex = 0; clusterIndex < clusters.size(); ++clusterIndex) {
        Vec3b centroidColor = Vec3b(
            static_cast<uchar>(centroids[clusterIndex][0]),
            static_cast<uchar>(centroids[clusterIndex][1]),
            static_cast<uchar>(centroids[clusterIndex][2])
        );

        for (const auto& pixel : clusters[clusterIndex]) {
            quantizedImage.at<Vec3b>(pixel.first) = centroidColor;
        }
    }

    return quantizedImage;
}

int main(int argc, char* argv[]) {
    
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <image_path> <max_k> <sensitivity>\n";
        cerr << "Please provide an image path, max number of clusters (max_k), and sensitivity.\n";
        return 1;
    }

    Mat image = imread(argv[1]);
    if (image.empty()) {
        cerr << "Could not open or find the image: " << argv[1] << "\n";
        return 1;
    }

    ColorToGrayConverter converter;

    int k = 1;
    int max_k = atoi(argv[2]);

    float sigma = 100; 
    if(argc >= 3){
        sigma = atof(argv[3]);
    }

    float theta_0 = 0.0004;  
    float theta_1 = 0.5; 
    
    converter.quantizeColors(image, k, max_k, theta_0, theta_1);
    converter.ordering(image);
    converter.createGrayScale(image, sigma);
    
    return 0;
}
