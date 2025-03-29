//nvcc newGLTest.cu -o newGLTest -lGL -lglfw

// Include files
//#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <GLFW/glfw3.h>
//#include <glad/glad.h> no glad needed for GLFW alone

// Defines
#define PI 3.14159265359
#define DRAW_RATE 10
#define N 100

// This is to create a Lennard-Jones type function G/(r^p) - H(r^q). (p < q) p has to be less than q.
// In this code we will keep it a p = 2 and q = 4 problem. The diameter of a body is found using the general
// case so it will be more robust but in the code leaving it as a set 2, 4 problem make the coding much easier.
#define G 10.0
#define H 10.0
#define LJP 2.0
#define LJQ 4.0

// Globals
float3 *P, *V, *F;
float *M;
float GlobeRadius, Diameter, Radius;
float Damp;
int DrawCount;
float Dt;
GLFWwindow* Window;

// Need to add these globals for GLFW
float Near = 0.1f;
float Far = 100.0f;
float EyeX = 0.0f, EyeY = 0.0f, EyeZ = 10.0f;
float CenterX = 0.0f, CenterY = 0.0f, CenterZ = 0.0f;
float UpX = 0.0f, UpY = 1.0f, UpZ = 0.0f;
float3 BackGround = {0.0f, 0.0f, 0.0f};

// Function prototypes
void renderSphere(float radius, int slices, int stacks);
void setup();
void drawPicture();
void nBody(float dt);
int main(int, char**);

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
    
    DrawCount = 0;
    Dt = 0.0001;
    Damp = 0.5;
    
    M = (float*)malloc(N*sizeof(float));
    P = (float3*)malloc(N*sizeof(float3));
    V = (float3*)malloc(N*sizeof(float3));
    F = (float3*)malloc(N*sizeof(float3));
    
    // This is the value where the force is zero for the L-J type force.
    Diameter = pow(H/G, 1.0/(LJQ - LJP)); 
    Radius = Diameter/2.0;

    // Using the radius of a body and a 68% packing ratio to find the radius of a global sphere that should hold all the bodies.
    // Then we double this radius just so we can get all the bodies setup with no problems.
    float totalVolume = float(N)*(4.0/3.0)*PI*Radius*Radius*Radius;
    totalVolume /= 0.68;
    float totalRadius = pow(3.0*totalVolume/(4.0*PI), 1.0/3.0);
    GlobeRadius = 2.0*totalRadius;

    // Randomly setting these bodies in the global sphere and setting the initial velocity, initial force, and mass.
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
    printf("\n Setup is done.\n");
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
        renderSphere(Radius, 20, 20);
        glPopMatrix();
    }

    glfwSwapBuffers(Window);
}

void nBody(float dt)
{
    float force_mag;
    float dx, dy, dz, d, d2;

    // Reset forces
    for(int i=0; i<N; i++)
    {
        F[i].x = 0.0;
        F[i].y = 0.0;
        F[i].z = 0.0;
    }

    // Calculate forces between all pairs of bodies
    for(int i=0; i<N; i++)
    {
        for(int j=i+1; j<N; j++)
        {
            dx = P[j].x-P[i].x;
            dy = P[j].y-P[i].y;
            dz = P[j].z-P[i].z;
            d2 = dx*dx + dy*dy + dz*dz;
            d  = sqrt(d2);

            force_mag = (G*M[i]*M[j])/(d2) - (H*M[i]*M[j])/(d2*d2);
            F[i].x += force_mag*dx/d;
            F[j].x -= force_mag*dx/d;
            F[i].y += force_mag*dy/d;
            F[j].y -= force_mag*dy/d;
            F[i].z += force_mag*dz/d;
            F[j].z -= force_mag*dz/d;
        }
    }

    // Update velocities and positions
    for(int i=0; i<N; i++)
    {
        V[i].x += ((F[i].x-Damp*V[i].x)/M[i])*dt;
        V[i].y += ((F[i].y-Damp*V[i].y)/M[i])*dt;
        V[i].z += ((F[i].z-Damp*V[i].z)/M[i])*dt;

        P[i].x += V[i].x*dt;
        P[i].y += V[i].y*dt;
        P[i].z += V[i].z*dt;
    }

    if(DrawCount == DRAW_RATE)
    {
        drawPicture();
        DrawCount = 0;
    }

    DrawCount++;
}

int main(int argc, char** argv)
{
    setup();

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

    // Set the viewport size and aspect ratio
    glViewport(0, 0, XWindowSize, YWindowSize);

    // PROJECTION MATRIX
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-0.2, 0.2, -0.2, 0.2, Near, Far);

    // MODELVIEW MATRIX
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, -5.0); // Move the camera back (Replaces gluLookAt)

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

    // Main loop
    while (!glfwWindowShouldClose(Window))
    {
        // Poll events
        glfwPollEvents();

        // Update physics with fixed timestep
        nBody(Dt);

        // Render the scene
        drawPicture();

        // Swap buffers
        glfwSwapBuffers(Window);
    }

    // Cleanup and exit
    glfwDestroyWindow(Window);
    glfwTerminate();

    return 0;
}