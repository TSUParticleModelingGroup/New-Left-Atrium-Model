//nvcc newGLTest.cu -o newGLTest -lGL -lglfw -lGLEW

// Include files
//#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <GL/glew.h> // Needed for VBOs + CUDA, using GLAD in main code
#include <GLFW/glfw3.h>

//needed for VBO rendering
#include <vector>

//#include <glad/glad.h> no glad needed for GLFW alone

// Defines
#define PI 3.14159265359
#define DRAW_RATE 1
#define N 50000

// This is to create a Lennard-Jones type function G/(r^p) - H(r^q). (p < q) p has to be less than q.
// In this code we will keep it a p = 2 and q = 4 problem. The diameter of a body is found using the general
// case so it will be more robust but in the code leaving it as a set 2, 4 problem make the coding much easier.
#define G 10.0
#define H 10.0
#define LJP 2.0
#define LJQ 4.0

dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid

// Globals
float3 *P, *V, *F, *P_GPU, *V_GPU, *F_GPU;
float *M_GPU;
float *M;
float GlobeRadius, Diameter, Radius;
float Damp;
int DrawCount = 0;
float Dt;
GLFWwindow* Window;

//To use VBOs for sphere rendering
GLuint sphereVBO, sphereIBO; // Vertex Buffer Object and Index Buffer Object for sphere rendering, Vertex is the sphere's vertices and Index is the order in which to draw them
GLuint numSphereVertices, numSphereIndices; // Number of vertices and indices in the sphere geometry

// Need to add these globals for GLFW
float Near = 0.1f;
float Far = 100.0f;
float EyeX = 0.0f, EyeY = 0.0f, EyeZ = 10.0f;
float CenterX = 0.0f, CenterY = 0.0f, CenterZ = 0.0f;
float UpX = 0.0f, UpY = 1.0f, UpZ = 0.0f;
float3 BackGround = {0.0f, 0.0f, 0.0f};

// Function prototypes
void cudaErrorCheck(const char *file, int line);
void renderSphere(float radius, int slices, int stacks);
void setup();
__global__ void getForces(float3 *P, float3 *V, float3 *F, float *M, int n);
__global__ void updatePositions(float3 *P, float3 *V, float3 *F, float *M, int n, float dt, float damp, float time);
void drawPicture();
void nBody(float dt);
int main(int, char**);

void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

// Function to render a sphere using VBOs
void createSphereVBO(float radius, int slices, int stacks) //this one builds a single sphere VBO that we can use for rendering each body in the simulation.
{
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    
    // Generate sphere vertices with positions and normals
    for (int i = 0; i <= stacks; ++i) 
    {
        float phi = PI * i / stacks;
        float sinPhi = sin(phi);
        float cosPhi = cos(phi);
        
        for (int j = 0; j <= slices; ++j) 
        {
            float theta = 2.0f * PI * j / slices;
            float sinTheta = sin(theta);
            float cosTheta = cos(theta);
            
            // Vertex position (x, y, z)
            float x = radius * sinPhi * cosTheta;
            float y = radius * cosPhi;
            float z = radius * sinPhi * sinTheta;
            
            // Normal vector (normalized position for sphere)
            float nx = sinPhi * cosTheta;
            float ny = cosPhi;
            float nz = sinPhi * sinTheta;
            
            // Add vertex (position + normal)
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
            vertices.push_back(nx);
            vertices.push_back(ny);
            vertices.push_back(nz);
        }
    }
    
    // Generate indices for triangle strips
    for (int i = 0; i < stacks; ++i) 
    {
        for (int j = 0; j < slices; ++j) 
        {
            int first = i * (slices + 1) + j;
            int second = first + slices + 1;
            
            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);
            
            indices.push_back(second);
            indices.push_back(second + 1);
            indices.push_back(first + 1);
        }
    }
    
    numSphereVertices = vertices.size() / 6; // 6 floats per vertex (pos + normal)
    numSphereIndices = indices.size();
    
    // Create and bind the vertex buffer
    glGenBuffers(1, &sphereVBO);
    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    
    // Create and bind the index buffer
    glGenBuffers(1, &sphereIBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
    
    // Unbind buffers
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void renderSphereVBO() 
{
    // Bind the VBO and IBO
    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereIBO);
    
    // Enable vertex and normal arrays
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    
    // Set up pointers to vertex and normal data
    glVertexPointer(3, GL_FLOAT, 6 * sizeof(float), 0);
    glNormalPointer(GL_FLOAT, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    
    // Draw the sphere
    glDrawElements(GL_TRIANGLES, numSphereIndices, GL_UNSIGNED_INT, 0);
    
    // Disable arrays
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    
    // Unbind buffers
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}


void renderSphere(float radius, int slices, int stacks) 
{
    // Sphere geometry parameters
    float x, y, z, alpha, beta; // Storage for coordinates and angles
    float sliceStep = 2.0f * PI / slices;
    float stackStep = PI / stacks;

    for (int i = 0; i < stacks; ++i) {
        alpha = i * stackStep;
        beta = alpha + stackStep;

        glBegin(GL_TRIANGLE_STRIP);
        for (int j = 0; j <= slices; ++j) {
            float theta = (j == slices) ? 0.0f : j * sliceStep;

            // Vertex 1
            x = -sin(alpha) * cos(theta);
            y = cos(alpha);
            z = sin(alpha) * sin(theta);
            glNormal3f(x, y, z);
            glVertex3f(x * radius, y * radius, z * radius);

            // Vertex 2
            x = -sin(beta) * cos(theta);
            y = cos(beta);
            z = sin(beta) * sin(theta);
            glNormal3f(x, y, z);
            glVertex3f(x * radius, y * radius, z * radius);
        }
        glEnd();
    }
}

void setup()
{
    float randomAngle1, randomAngle2, randomRadius;
    float d, dx, dy, dz;
    int test;

    // Set the block and grid sizes
    BlockSize.x = 1024;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = (N - 1)/BlockSize.x + 1;
	GridSize.y = 1;
	GridSize.z = 1;


    Diameter = pow(H/G, 1.0/(LJQ - LJP)); // This is the value where the force is zero for the L-J type force.
	Radius = Diameter/2.0;
    
    

    Damp = 0.5;
    
    M = (float*)malloc(N*sizeof(float));
    P = (float3*)malloc(N*sizeof(float3));
    V = (float3*)malloc(N*sizeof(float3));
    F = (float3*)malloc(N*sizeof(float3));

    // Allocate memory on the GPU, doing it here so all mallocs are in one place.
    cudaMalloc(&P_GPU, N * sizeof(float3));
    cudaErrorCheck(__FILE__, __LINE__);

    cudaMalloc(&V_GPU, N * sizeof(float3));
    cudaErrorCheck(__FILE__, __LINE__);

    cudaMalloc(&F_GPU, N * sizeof(float3));
    cudaErrorCheck(__FILE__, __LINE__);

    cudaMalloc(&M_GPU, N * sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
	
	// Using the radius of a body and a 68% packing ratio to find the radius of a global sphere that should hold all the bodies.
	// Then we double this radius just so we can get all the bodies setup with no problems. 
	float totalVolume = float(N)*(4.0/3.0)*PI*Radius*Radius*Radius;
	totalVolume /= 0.68;
	float totalRadius = pow(3.0*totalVolume/(4.0*PI), 1.0/3.0);
	GlobeRadius = 2.0*totalRadius;
	
	// Randomly setting these bodies in the glaobal sphere and setting the initial velosity, inotial force, and mass.
	for(int i = 0; i < N; i++)
	{
		test = 0;
		while(test == 0)
		{
			// Get random position.
			randomAngle1 = ((float)rand()/(float)RAND_MAX)*2.0*PI;
			randomAngle2 = ((float)rand()/(float)RAND_MAX)*PI;
			randomRadius = ((float)rand()/(float)RAND_MAX)*GlobeRadius;
			P[i].x = randomRadius*cos(randomAngle1)*sin(randomAngle2);
			P[i].y = randomRadius*sin(randomAngle1)*sin(randomAngle2);
			P[i].z = randomRadius*cos(randomAngle2);
			
			// Making sure the balls centers are at least a diameter apart.
			// If they are not throw these positions away and try again.
			test = 1;
			for(int j = 0; j < i; j++)
			{
				dx = P[i].x-P[j].x;
				dy = P[i].y-P[j].y;
				dz = P[i].z-P[j].z;
				d = sqrt(dx*dx + dy*dy + dz*dz);
				if(d < Diameter)
				{
					test = 0;
					break;
				}
			}
		}
	
		V[i].x = 0.0;
		V[i].y = 0.0;
		V[i].z = 0.0;
		
		F[i].x = 0.0;
		F[i].y = 0.0;
		F[i].z = 0.0;
		
		M[i] = 1.0;
	}
}

__global__ void leapFrog(float3 *p, float3 *v, float3 *f, float *m, float g, float h, float damp, float dt, float t, int n)
{
	float dx, dy, dz,d,d2;
	float force_mag;
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(i < n)
	{
		f[i].x = 0.0f;
		f[i].y = 0.0f;
		f[i].z = 0.0f;

		for(int j = 0; j < n; j++)
		{
			if(i != j)
			{
				dx = p[j].x-p[i].x;
				dy = p[j].y-p[i].y;
				dz = p[j].z-p[i].z;
				d2 = dx*dx + dy*dy + dz*dz;
				d  = sqrt(d2);
				
				force_mag  = (g*m[i]*m[j])/(d2) - (h*m[i]*m[j])/(d2*d2);
				f[i].x += force_mag*dx/d * 1000.0f;
				f[i].y += force_mag*dy/d * 1000.0f;
				f[i].z += force_mag*dz/d * 1000.0f;
			}
		}
		__syncthreads();
		
		if(t == 0.0f)
		{
			v[i].x += ((f[i].x-damp*v[i].x)/m[i])*dt/2.0f;
			v[i].y += ((f[i].y-damp*v[i].y)/m[i])*dt/2.0f;
			v[i].z += ((f[i].z-damp*v[i].z)/m[i])*dt/2.0f;
		}
		else
		{
			v[i].x += ((f[i].x-damp*v[i].x)/m[i])*dt;
			v[i].y += ((f[i].y-damp*v[i].y)/m[i])*dt;
			v[i].z += ((f[i].z-damp*v[i].z)/m[i])*dt;
		}

		p[i].x += v[i].x*dt;
		p[i].y += v[i].y*dt;
		p[i].z += v[i].z*dt;
		__syncthreads();
	}
	
}

void drawPicture()
{
    int i;

    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);

    
    for(i=0; i<N; i++)
    {
        glColor3d(1.0, 1.0, 0.5);

        glPushMatrix();
        glTranslatef(P[i].x, P[i].y, P[i].z);
        //renderSphere(Radius, 20, 20);
        renderSphereVBO(); // Use VBO for rendering instead of the above function, uncomment this line to use VBOs instead of the custom renderSphere function
        glPopMatrix();
    }

    //glfwSwapBuffers(Window);
}

void nBody(float dt)
{
	float  time = 0.0;

    leapFrog<<<GridSize, BlockSize>>>(P_GPU, V_GPU, F_GPU, M_GPU, G, H, Damp, dt, time, N);
    cudaDeviceSynchronize();

    //draw if we need to
    if(DrawCount == DRAW_RATE) 
    {
        cudaMemcpy(P, P_GPU, N * sizeof(float3), cudaMemcpyDeviceToHost); //only copy pos to CPU if drawing
        cudaErrorCheck(__FILE__, __LINE__);
        drawPicture();
        DrawCount = 0;
    }
    
    time += dt;
    DrawCount++;



}

int main(int argc, char** argv)
{
    setup();

	// cudaMemcpy(P, P_GPU, N * sizeof(float3), cudaMemcpyDeviceToHost);
	// cudaErrorCheck(__FILE__, __LINE__);

	// cudaMemcpy(V, V_GPU, N * sizeof(float3), cudaMemcpyDeviceToHost);
	// cudaErrorCheck(__FILE__, __LINE__);

	// cudaMemcpy(F, F_GPU, N * sizeof(float3), cudaMemcpyDeviceToHost);
	// cudaErrorCheck(__FILE__, __LINE__);

    int XWindowSize = 1000;
    int YWindowSize = 1000;

    if (!glfwInit()) // Initialize GLFW, check for failure
    {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    // Set compatibility mode to allow legacy OpenGL
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);

    // Create a windowed mode window and its OpenGL context
    Window = glfwCreateWindow(XWindowSize, YWindowSize, "N-Body Simulation", NULL, NULL);
    if (!Window)
    {
        glfwTerminate();
        fprintf(stderr, "Failed to create window\n");
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(Window);
    glfwSwapInterval(1); // Enable vsync

    //initialize GLEW
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "GLEW initialization error: %s\n", glewGetErrorString(err));
        glfwTerminate();
        return -1;
    }

    //initialize the sphere VBO for rendering --MUST BE DONE AFTER GLEW/GLAD INIT
    createSphereVBO(Radius, 20, 20); 

    // Set the viewport size and aspect ratio
    glViewport(0, 0, XWindowSize, YWindowSize);

    // PROJECTION MATRIX
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-0.2, 0.2, -0.2, 0.2, Near, Far);

    // MODELVIEW MATRIX
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, -0.001*N); // Move the camera back (Replaces gluLookAt)

    glClearColor(BackGround.x, BackGround.y, BackGround.z, 0.0);

    // Lighting and material properties
    glEnable(GL_LIGHTING); // Enable lighting
    glEnable(GL_LIGHT0);   // Enable light source 0

    // Configure light properties
    GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0}; // Directional light
    GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0}; // Ambient light (dim)
    GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0}; // Diffuse light (bright white)
    GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0}; // Specular light (white)

    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);

    // Configure global ambient light
    GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0}; // Global ambient light
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);

    // Enable color material to use glColor for ambient and diffuse material properties
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);

    // Set default material properties
    GLfloat mat_specular[] = {1.0, 1.0, 1.0, 1.0}; // Specular reflection
    GLfloat mat_shininess[] = {10.0};             // Shininess factor
    glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
    glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);

    // Enable depth testing for proper 3D rendering
    glEnable(GL_DEPTH_TEST);

    // Time variables
    float Dt = 0.0001;
    
    //copy before we start the main loop to ensure we have the initial values on the GPU
    cudaMemcpy(P_GPU, P, N * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(V_GPU, V, N * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(F_GPU, F, N * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(M_GPU, M, N * sizeof(float), cudaMemcpyHostToDevice);

    // Main loop
    while (!glfwWindowShouldClose(Window))
    {
        // Poll events
        glfwPollEvents();

        // Update physics with fixed timestep
        nBody(Dt);

        // Render the scene
        //drawPicture();

        // Swap buffers
        glfwSwapBuffers(Window);
    }

    // Cleanup and exit
    glfwDestroyWindow(Window);
    glfwTerminate();

    return 0;
}