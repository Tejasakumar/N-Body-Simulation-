#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <GLFW/glfw3.h>
#include<windows.h> 
#include <vector>
#include <random>

using namespace std;

struct Body {
    float3 position;
    float3 velocity;
    float mass;
    GLfloat color[3];
};

#define G 6.67e-11
#define epsilon 1e-11
const int width = 800;
const int height = 600;
const int NUMBODIES = 1000;


int generateSeed() {
    return static_cast<int>(time(NULL));
}

__global__ 
void noramlize(Body* arr,Body *dest) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const float min_mass = 3.3011e23;
    const float max_mass = 1.989e30;
    const float min_normalized_mass = 0.005;
    const float max_normalized_mass = 0.015;

    for (int i = index; i < NUMBODIES; i += stride) {
        // Normalize position coordinates to NDC range [-1, 1]
        dest[i].position.x =arr[i].position.x / 1.0e11; // Divide by max value to get [-1, 1] range
        dest[i].position.y =arr[i].position.y / 1.0e11;
        dest[i].position.z =arr[i].position.z / 1.0e11;
        for (int j = 0; j < 3; ++j) {
            dest[i].color[j] = arr[i].color[j];
        }
        float original_mass = arr[i].mass;
        float normalized_mass = (original_mass - min_mass) / (max_mass - min_mass)
            * (max_normalized_mass - min_normalized_mass) + min_normalized_mass;

        dest[i].mass = normalized_mass;
    }
}

__global__
void initialize(Body* deviceBodies, int num_bodies, int seed) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;


    GLfloat col1[] = { 225.0f / 255.0f, 111.0f / 255.0f, 103.0f / 255.0f };
    GLfloat col2[] = { 133.0f / 255.0f, 191.0f / 255.0f, 177.0f / 255.0f };
    GLfloat col3[] = { 138.0f / 255.0f, 190.0f / 255.0f, 178.0f / 255.0f };
    GLfloat col4[] = { 241.0f / 255.0f, 221.0f / 255.0f, 182.0f / 255.0f };

    GLfloat* cols[] = { col1, col2, col3, col4 };

    
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < num_bodies; i += stride) {

        curandState state;
        curand_init(seed, i, 0, &state); // seed, sequence, offset, state

        deviceBodies[i].position.x = curand_uniform(&state) * 2.0e11 - 1.0e11; // Range: -1.0e11 to 1.0e11 meters
        deviceBodies[i].position.y = curand_uniform(&state) * 2.0e11 - 1.0e11; // Range: -1.0e11 to 1.0e11 meters
        deviceBodies[i].position.z = curand_uniform(&state) * 2.0e11 - 1.0e11; // Range: -1.0e11 to 1.0e11 meters


        deviceBodies[i].velocity.x = curand_uniform(&state) * (100000.0 - 15000.0) + 15000.0; // Range: 15,000 m/s to 100,000 m/s
        deviceBodies[i].velocity.y = curand_uniform(&state) * (100000.0 - 15000.0) + 15000.0; // Range: 15,000 m/s to 100,000 m/s
        deviceBodies[i].velocity.z = curand_uniform(&state) * (100000.0 - 15000.0) + 15000.0; // Range: 15,000 m/s to 100,000 m/s

        //deviceBodies[i].velocity.x = 0;
        //deviceBodies[i].velocity.y = 0;
        //deviceBodies[i].velocity.z = 0;

        deviceBodies[i].mass = curand_uniform(&state) * (1.989e30 - 3.3011e23) + 3.3011e23;
        //float log_normal_value = curand_log_normal(&state, 1.0f, 1.0f);
        //float mass = expf(log_normal_value);
        //float gaussian_value = curand_normal(&state);
        //float mass = gaussian_value * (1.989e30 - 3.3011e23) + 3.3011e23;
        //deviceBodies[i].mass = mass;

        int colIndex = i % (sizeof(cols) / sizeof(cols[0]));
        for (int j = 0; j < 3; ++j) {
            deviceBodies[i].color[j] = cols[colIndex][j];

        }
        //printf("%f %f %f\n", deviceBodies[i].color[0], deviceBodies[i].color[1], deviceBodies[i].color[2]);


    }
}



__global__ void computeAccn(Body* bodies, int num_bodies, float3* accelerations, float dt) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_bodies; i += stride) {
        float3 totalforce = { 0.0f,0.0f,0.0f };
        for (int j = 0; j < num_bodies; j++) {
            if (j != i) {
                float dist_x = bodies[j].position.x - bodies[i].position.x;
                float dist_y = bodies[j].position.y - bodies[i].position.y;
                float dist_z = bodies[j].position.z - bodies[i].position.z;

                float3 dist_vector = make_float3(dist_x, dist_y, dist_z);
                float dist_sq = dist_vector.x * dist_vector.x + dist_vector.y * dist_vector.y + dist_vector.z * dist_vector.z;
                float dist_cubed = sqrt(dist_sq) * sqrt(dist_sq) * sqrt(dist_sq);


                /*
                  Fij=(Gm1m2/(r^2+e^2)^3/2)rij

                */

                //printf("mass1 %f mass2 %f\n",bodies[i].mass,bodies[j].mass);
                float force_x = ((G * bodies[i].mass * bodies[j].mass) / dist_cubed) * dist_x;
                //printf("Distance %f\n", dist_x);
                //printf("Force %f\n",force_x);
                float force_y = G * bodies[i].mass * bodies[j].mass / dist_cubed * dist_y;
                float force_z = G * bodies[i].mass * bodies[j].mass / dist_cubed * dist_z;

                totalforce.x += force_x;
                totalforce.y += force_y;
                totalforce.z += force_z;


                //printf("Body %d DistCubed: %f\n", i, dist_cubed);
                //printf("Body %d Force: (%f, %f, %f)\n", i, totalforce.x, totalforce.y, totalforce.z);



            }

        }



        accelerations[i].x = totalforce.x / bodies[i].mass;
        accelerations[i].y = totalforce.y / bodies[i].mass;
        accelerations[i].z = totalforce.z / bodies[i].mass;
        //printf("Body %d Acceleration: (%f, %f, %f)\n", i, accelerations[i].x, accelerations[i].y, accelerations[i].z);
        bodies[i].velocity.x += accelerations[i].x * dt; //integral technically
        bodies[i].velocity.y += accelerations[i].y * dt;
        bodies[i].velocity.z += accelerations[i].z * dt;

        bodies[i].position.x += bodies[i].velocity.x * dt;
        bodies[i].position.y += bodies[i].velocity.y * dt;
        bodies[i].position.z += bodies[i].velocity.z * dt;
    }
}



// Function to display OpenGL scene
void displayy(GLFWwindow* window, Body* points, int numPoints) {
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(1.0, 1.0, 1.0); // White color

    // Draw points
    glBegin(GL_POINTS);
    for (int i = 0; i < numPoints; i++) {
        glVertex2f(points[i].position.x, points[i].position.y);
    }
    glEnd();

    glfwSwapBuffers(window);
}

// Function to display OpenGL scene
void display(GLFWwindow* window, Body* points, int numPoints) {

    

    glClear(GL_COLOR_BUFFER_BIT);
    //GLfloat color[] = { 1.0f, 1.0f, 1.0f }; // Array representing white color (RGB: 1.0, 1.0, 1.0)
    //glColor3fv(color); // Pass the array to glColor3fv


    // Draw circles
    float radius = 0.005f; // Set radius for circles
    for (int i = 0; i < numPoints; i++) {

        GLfloat color[] = { points[i].color[0], points[i].color[1], points[i].color[2] };
        glColor3fv(color);
        //cout << points[i].color[0] << " " << points[i].color[1] << " " << points[i].color[2] << endl;

        radius = points[i].mass;
        
        glBegin(GL_TRIANGLE_FAN);
        glVertex2f(points[i].position.x, points[i].position.y); // Center of circle
        for (int j = 0; j <= 360; j += 10) { // Draw circle using triangle fan
            float angle = j * 3.14159f / 180.0f;
            float x = points[i].position.x + radius * cos(angle);
            float y = points[i].position.y + radius * sin(angle);
            glVertex2f(x, y);
        }
        glEnd();
    }

    glfwSwapBuffers(window);
}


int main() {
    // Initialize GLFW
    
    if (!glfwInit()) {
        cerr << "Failed to initialize GLFW" << endl;
        return -1;
    }

    // Create a windowed mode window and its OpenGL context
    GLFWwindow* window = glfwCreateWindow(width, height, "CUDA GLFW OpenGL Random Points", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Initialize CUDA
    cudaSetDevice(0);

    



    int threads_per_block = 256;
    int blocks_per_grid = (NUMBODIES - 1 + threads_per_block) / threads_per_block;
    int seed = generateSeed();
    Body* deviceBodies;
    cudaMalloc((void**)&deviceBodies, NUMBODIES * sizeof(Body));

    initialize <<< blocks_per_grid, threads_per_block >>> (deviceBodies, NUMBODIES, seed);
    cudaDeviceSynchronize();

    // Allocate memory for points on the host
    Body* points = new Body[NUMBODIES * 2];

    // Copy points from device to host
    cudaMemcpy(points, deviceBodies, sizeof(float) * NUMBODIES * 2, cudaMemcpyDeviceToHost);

    


    Body* normalized_bodies = (Body*)malloc(sizeof(Body) * NUMBODIES);
    Body* d_norm_bodies;
    cudaMalloc((void**)&d_norm_bodies, NUMBODIES * sizeof(Body));

    noramlize <<<blocks_per_grid, threads_per_block >> > (deviceBodies,d_norm_bodies);

    cudaMemcpy(normalized_bodies, d_norm_bodies, sizeof(float) * NUMBODIES * 2, cudaMemcpyDeviceToHost);

    cout << normalized_bodies[0].color[0] << " " << normalized_bodies[0].color[1] << " " << normalized_bodies[0].color[2] << endl;


    float3* deviceAccelerations;


    float3* hostAccelerations = (float3*)malloc(sizeof(float3) * NUMBODIES);

    cudaMalloc((void**)&deviceAccelerations, NUMBODIES * sizeof(float3));


    float dt = 500;
    int frame_cnt = 0;

    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window) && frame_cnt!=1000) {
        // Render here
        display(window, normalized_bodies, NUMBODIES);
        computeAccn <<< blocks_per_grid, threads_per_block >>> (deviceBodies, NUMBODIES, deviceAccelerations, dt);

        cudaMemcpy(points, deviceBodies, NUMBODIES, cudaMemcpyDeviceToHost);
        noramlize << <blocks_per_grid, threads_per_block >> > (deviceBodies, d_norm_bodies);
        cudaMemcpy(normalized_bodies, d_norm_bodies, sizeof(float) * NUMBODIES * 2, cudaMemcpyDeviceToHost);

        //generateRandomPoints << <(numPoints + 255) / 256, 256 >> > (d_points, time(NULL));
        //cudaMemcpy(points, d_points, sizeof(float) * numPoints * 2, cudaMemcpyDeviceToHost);

        //Sleep(500);
        // Poll for and process events
        cout << "Frame no= " << frame_cnt++ <<  endl;
        glfwPollEvents();
        
    }

    // Free memory
    delete[] points;
    cudaFree(deviceBodies);
    cudaFree(d_norm_bodies);
    free(normalized_bodies);
    
    // Terminate GLFW
    glfwTerminate();

    return 0;
}