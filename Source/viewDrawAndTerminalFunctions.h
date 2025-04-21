/*
 This file contains:
 1: All the functions that determine how to orient and view the simulation.
 2: all the functions that draw the actual simulation. 
 3: The functions that print to the linux terminal all the setting of the simulation.
 In short this file holds the functions that present information to the user.
 
 The functions are listed below in the order they appear.
 void renderSphere(float, int, int);
 void createSphereVBO(float, int, int);
 void renderSphereVBO();
 void orthogonalView();
 void frustumView();
 float4 findCenterOfMass();
 void centerObject();
 void rotateXAxis(float);
 void rotateYAxis(float);
 void rotateZAxis(float);
 void ReferenceView();
 void PAView();
 void APView();
 void setView(int);
 void drawPicture();
 void createGUI();
*/

/*
 This function sets your view to orthogonal. In orthogonal view all object are kept in line in the z direction.
 This is not how your eye sees things but can be useful when determining if objects are lined up along the z-axis. 
*/

// Add this to a utility file, only used for the mouse selection since it's just 1 object
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
/*
	Function to render a sphere using a VBO
	This function creates a VBO for a sphere and binds it for rendering.

	This code creates vertices and indices to make a sphere using triangle strips.
	It uses OpenGL functions to create and bind the VBO and IBO (what makes up the sphere).
	The sphere stays in the GPU memory and is faster to render and puts less load on the CPU.

*/
void createSphereVBO(float radius, int slices, int stacks)
{
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    
    // Generate sphere vertices with positions and normals
	for (int i = 0; i <= stacks; ++i) 
	{
		// Calculate the vertical angle phi (0 to PI, from top to bottom of sphere)
		float phi = PI * i / stacks;
		float sinPhi = sin(phi);
		float cosPhi = cos(phi);
		
		for (int j = 0; j <= slices; ++j) 
		{
			// Calculate the horizontal angle theta (0 to 2PI, around the sphere)
			float theta = 2.0f * PI * j / slices;
			float sinTheta = sin(theta);
			float cosTheta = cos(theta);
			
			// Convert spherical to Cartesian coordinates
			// x = r * sin(phi) * cos(theta)
			// y = r * cos(phi)          // y is up/down axis (poles of the sphere)
			// z = r * sin(phi) * sin(theta)
			float x = radius * sinPhi * cosTheta;
			float y = radius * cosPhi;
			float z = radius * sinPhi * sinTheta;
			
			// For a sphere, normal vectors point outward from center
			// and are simply the normalized position vector (position/radius)
			float nx = sinPhi * cosTheta;  // Same as x/radius
			float ny = cosPhi;             // Same as y/radius
			float nz = sinPhi * sinTheta;  // Same as z/radius
			
			// Store the vertex data in interleaved format:
			// Each vertex has 6 floats - 3 for position (x,y,z) and 3 for normal (nx,ny,nz)
			vertices.push_back(x);
			vertices.push_back(y);
			vertices.push_back(z);
			vertices.push_back(nx);
			vertices.push_back(ny);
			vertices.push_back(nz);
		}
	}
    
	// Generate indices for triangle strips
	// This section creates triangles by connecting the grid of vertices:
	// - First defines index values that point to positions in the vertex array 
	// - Creates two triangles for each grid cell (rectangular patch)
	// - Each triangle is defined by three indices in counter-clockwise order
	for (int i = 0; i < stacks; ++i) 
	{
		for (int j = 0; j < slices; ++j) 
		{
			// Calculate indices for the four corners of the current grid cell
			int first = i * (slices + 1) + j;          // Current vertex
			int second = first + slices + 1;           // Vertex below current
			
			// First triangle: Connect current vertex, vertex below, and vertex to the right
			indices.push_back(first);
			indices.push_back(second);
			indices.push_back(first + 1);
			
			// Second triangle: Connect vertex below, vertex below+right, and vertex to the right
			indices.push_back(second);
			indices.push_back(second + 1);
			indices.push_back(first + 1);
		}
	}

	// Store the total counts for rendering
	numSphereVertices = vertices.size() / 6; // 6 floats per vertex (pos + normal)
	numSphereIndices = indices.size();

	// Create and setup OpenGL buffers on the GPU
	// - Generate unique buffer IDs
	// - Bind buffers to set them as active
	// - Copy data from CPU arrays to GPU memory
	glGenBuffers(1, &sphereVBO);  // Generate Vertex Buffer Object for storing positions and normals
	glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

	// Same process for the index buffer
	glGenBuffers(1, &sphereIBO);  // Generate Index Buffer Object for storing triangle connections
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereIBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

	// Unbind buffers to prevent accidental modification
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

void orthogonalView()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-RadiusOfLeftAtrium, RadiusOfLeftAtrium, -RadiusOfLeftAtrium, RadiusOfLeftAtrium, Near, Far);
	glMatrixMode(GL_MODELVIEW);
	Simulation.ViewFlag = 0;
	drawPicture();
}

/*
 This function sets your view to frustum.This is the view the your eyes actually see. Where train tracks pull in 
 towards each other as they move off in the distance. It is how we see but can cause problems when using the mouse
 which lives in 2D to locate an object that lives in 3D.
*/
void frustumView()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, Near, Far);
	glMatrixMode(GL_MODELVIEW);
	Simulation.ViewFlag = 1;
	drawPicture();
}

/*
 This function finds the center of mass of the LA. It may seem like this function does not belong with
 the display functions but it is used to center th LA in the view.
*/
float4 findCenterOfMass()
{
	float4 centerOfMass;
	
	centerOfMass.x = 0.0;
	centerOfMass.y = 0.0;
	centerOfMass.z = 0.0;
	centerOfMass.w = 0.0;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		 centerOfMass.x += Node[i].position.x*Node[i].mass;
		 centerOfMass.y += Node[i].position.y*Node[i].mass;
		 centerOfMass.z += Node[i].position.z*Node[i].mass;
		 centerOfMass.w += Node[i].mass;
	}
	if(centerOfMass.w < 0.00001) // .w holds the mass.
	{
		printf("\n Mass is too small\n");
		printf("\nw Good Bye\n");
		exit(0);
	}
	else
	{
		centerOfMass.x /= centerOfMass.w;
		centerOfMass.y /= centerOfMass.w;
		centerOfMass.z /= centerOfMass.w;
	}
	return(centerOfMass);
}

/*
 This function centers the LA and resets the center of view to (0, 0, 0).
 It is called periodically in a running simulation to center the LA, because the LA is not symmetrical 
 and will wander off over time. It is also use center the LA before all the views are set.
*/
void centerObject()
{
	float4 centerOfMass = findCenterOfMass();
	for(int i = 0; i < NumberOfNodes; i++)
	{
		Node[i].position.x -= centerOfMass.x;
		Node[i].position.y -= centerOfMass.y;
		Node[i].position.z -= centerOfMass.z;
		
		//Node[i].velocity.x = 0.0;
		//Node[i].velocity.y = 0.0;
		//Node[i].velocity.z = 0.0;
	}
	CenterOfSimulation.x = 0.0;
	CenterOfSimulation.y = 0.0;
	CenterOfSimulation.z = 0.0;
}

/*
 This function rotates the view around the x-axis.
*/
void rotateXAxis(float angle)
{
	float temp;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		temp = cos(angle)*Node[i].position.y - sin(angle)*Node[i].position.z;
		Node[i].position.z  = sin(angle)*Node[i].position.y + cos(angle)*Node[i].position.z;
		Node[i].position.y  = temp;
	}
	AngleOfSimulation.x += angle;
}

/*
 This function rotates the view around the y-axis.
*/
void rotateYAxis(float angle)
{
	float temp;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		temp =  cos(-angle)*Node[i].position.x + sin(-angle)*Node[i].position.z;
		Node[i].position.z  = -sin(-angle)*Node[i].position.x + cos(-angle)*Node[i].position.z;
		Node[i].position.x  = temp;
	}
	AngleOfSimulation.y += angle;
}

/*
 This function rotates the view around the z-axis.
*/
void rotateZAxis(float angle)
{
	float temp;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		temp = cos(angle)*Node[i].position.x - sin(angle)*Node[i].position.y;
		Node[i].position.y  = sin(angle)*Node[i].position.x + cos(angle)*Node[i].position.y;
		Node[i].position.x  = temp;
	}
	AngleOfSimulation.z += angle;
}

/*
 This function puts the viewer in the reference view. The reference view is looking straight at the four
 pulmonary veins with a vein in each of the four quadrants of the x-y plane as symmetric as you can make 
 it with the mitral valve down. We base all the other views off of this view.
*/
void ReferenceView()
{	
	float angle, temp;
	
	centerObject();
		
	// Rotating until the up Node is on x-y plane above or below the positive x-axis.
	angle = atan(Node[UpNode].position.z/Node[UpNode].position.x);
	if(Node[UpNode].position.x < 0.0) angle -= PI;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		temp = cos(angle)*Node[i].position.x + sin(angle)*Node[i].position.z;
		Node[i].position.z  = -sin(angle)*Node[i].position.x + cos(angle)*Node[i].position.z;
		Node[i].position.x  = temp;
	}
	AngleOfSimulation.y += angle;
	
	// Rotating until up Node is on the positive y axis.
	angle = PI/2.0 - atan(Node[UpNode].position.y/Node[UpNode].position.x);
	for(int i = 0; i < NumberOfNodes; i++)
	{
		temp = cos(angle)*Node[i].position.x - sin(angle)*Node[i].position.y;
		Node[i].position.y  = sin(angle)*Node[i].position.x + cos(angle)*Node[i].position.y;
		Node[i].position.x  = temp;
	}
	AngleOfSimulation.z += angle;
	
	// Rotating until front Node is on the positive z axis.
	angle = atan(Node[FrontNode].position.z/Node[FrontNode].position.x) - PI/2.0;
	if(Node[FrontNode].position.x < 0.0) angle -= PI;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		temp = cos(angle)*Node[i].position.x + sin(angle)*Node[i].position.z;
		Node[i].position.z  = -sin(angle)*Node[i].position.x + cos(angle)*Node[i].position.z;
		Node[i].position.x  = temp;
	}
	AngleOfSimulation.y += angle;
}

/*
 This function puts the LA in the PA view.
 The heart does not set in the chest at a straight on angle. Hence we need to adjust our 
 reference view to what is actually seen in a back view looking through the chest.
*/
void PAView()
{  
	float angle;
	
	ReferenceView();
	
	angle = PI/6.0; // Rotate 30 degrees counterclockwise on the y-axis 
	rotateYAxis(angle);

	angle = PI/6.0; // Rotate 30 degrees counterclockwise on the z-axis
	rotateZAxis(angle);
}

/*
 This function puts the LA in the AP view.
 To get the AP view we just rotate the PA view 180 degrees on the y-axis
*/
void APView()
{ 
	float angle;
	
	PAView();
	angle = PI; // Rotate 180 degrees counterclockwise on the y-axis 
	rotateYAxis(angle);
}

/*
 This function sets all the views based off of the reference view and the AP view.
*/
void setView(int view)
{
    if(view == 6)
    {
        ReferenceView();
        strcpy(ViewName, "Ref");
    }
    else if(view == 4)
    {
        PAView();
        strcpy(ViewName, "PA");
    }
    else if(view == 2)
    {
        APView();
        strcpy(ViewName, "AP");
    }
    else if(view == 3)
    {
        APView();
        rotateYAxis(-PI/6.0);
        strcpy(ViewName, "RAO");
    }
    else if(view == 1)
    {
        APView();
        rotateYAxis(PI/3.0);
        strcpy(ViewName, "LAO");
    }
    else if(view == 7)
    {
        APView();
        rotateYAxis(PI/2.0);
        strcpy(ViewName, "LL");
    }
    else if(view == 9)
    {
        APView();
        rotateYAxis(-PI/2.0);
        strcpy(ViewName, "RL");
    }
    else if(view == 8)
    {
        APView();
        rotateXAxis(PI/2.0);
        strcpy(ViewName, "SUP");
    }
    else if(view == 5)
    {
        APView();
        rotateXAxis(-PI/2.0);
        strcpy(ViewName, "INF");
    }
    else
    {
        printf("\n Undefined view reverting back to Ref view.");
    }
}

/*
 This function draws the LA to the screen. It also saves movie frames if a movie is being recorded.
*/
void drawPicture()
{
	//int nodeNumber;
	int muscleNumber;
	int k;
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	// Draw Nodes

	// Drawing Pulse node
	if(!Simulation.isPaused) glColor3d(0.0,1.0,0.0);
	else glColor3d(1.0,0.0,0.0);

	glPushMatrix();
	glTranslatef(Node[PulsePointNode].position.x, Node[PulsePointNode].position.y, Node[PulsePointNode].position.z);
	renderSphereVBO();
	glPopMatrix();
	
	// Drawing center node
	//This draws a center node at the center of the simulation for debugging purposes
	if(false) // false turns it off, true turns it on.
	{
		glColor3d(1.0,1.0,1.0);
		glPushMatrix();
		glTranslatef(CenterOfSimulation.x, CenterOfSimulation.y, CenterOfSimulation.z);
		renderSphereVBO();
		glPopMatrix();
	}

	// Drawing other nodes
	if(Simulation.DrawNodesFlag == 1 || Simulation.DrawNodesFlag == 2)  //if we're drawing half(1) or all(2) of the nodes
	{
		for(int i = 1; i < NumberOfNodes; i++) // Start at 1 to skip the pulse node and go through all nodes
		{
			if(Simulation.DrawFrontHalfFlag == 1 || Simulation.DrawNodesFlag == 1) //if we're only drawing the front half of the nodes
			{
				if(CenterOfSimulation.z - 0.001 < Node[i].position.z)  //draw only the nodes in the front.
				{
					glColor3d(Node[i].color.x, Node[i].color.y, Node[i].color.z);
					glPushMatrix();
					glTranslatef(Node[i].position.x, Node[i].position.y, Node[i].position.z);
					renderSphereVBO();
					glPopMatrix();
				}
			}
			else //draw all nodes
			{
				glColor3d(Node[i].color.x, Node[i].color.y, Node[i].color.z);
				glPushMatrix();
				glTranslatef(Node[i].position.x, Node[i].position.y, Node[i].position.z);
				renderSphereVBO();
				glPopMatrix();
			}	
		}
	}
	else
	{
		glPointSize(5.0);
		glBegin(GL_POINTS);
		 	for(int i = 1; i < NumberOfNodes; i++)
			{
				if(Simulation.DrawFrontHalfFlag == 1)
				{
					if(CenterOfSimulation.z - 0.001 < Node[i].position.z)  // Only drawing the nodes in the front.
					{
						glColor3d(Node[i].color.x, Node[i].color.y, Node[i].color.z);
						if(Node[i].isDrawNode)
						{
							glVertex3f(Node[i].position.x, Node[i].position.y, Node[i].position.z);
						}
					}
				}
				else
				{
					glColor3d(Node[i].color.x, Node[i].color.y, Node[i].color.z);
					if(Node[i].isDrawNode)
					{
						glVertex3f(Node[i].position.x, Node[i].position.y, Node[i].position.z);
					}
				}
			}
		glEnd();
	}
	
	// Drawing muscles
	glLineWidth(LineWidth);
	for(int i = 0; i < NumberOfNodes; i++)
	{
		for(int j = 0; j < MUSCLES_PER_NODE; j++)
		{
			muscleNumber = Node[i].muscle[j];
			if(muscleNumber != -1)
			{
				k = Muscle[muscleNumber].nodeA;
				if(k == i) 
				{
					k = Muscle[muscleNumber].nodeB;
				}
				
				if(Simulation.DrawFrontHalfFlag == 1)
				{
					if(CenterOfSimulation.z - 0.001 < Node[i].position.z && CenterOfSimulation.z - 0.001 < Node[k].position.z)  // Only drawing the nodes in the front.
					{
						glColor3d(Muscle[muscleNumber].color.x, Muscle[muscleNumber].color.y, Muscle[muscleNumber].color.z);
						glBegin(GL_LINES);
							glVertex3f(Node[i].position.x, Node[i].position.y, Node[i].position.z);
							glVertex3f(Node[k].position.x, Node[k].position.y, Node[k].position.z);
						glEnd();
					}
				}
				else
				{
					glColor3d(Muscle[muscleNumber].color.x, Muscle[muscleNumber].color.y, Muscle[muscleNumber].color.z);
					glBegin(GL_LINES);
						glVertex3f(Node[i].position.x, Node[i].position.y, Node[i].position.z);
						glVertex3f(Node[k].position.x, Node[k].position.y, Node[k].position.z);
					glEnd();
				}
			}
		}	
	}
	
	// Puts a ball at the location of the mouse if a mouse function is on.
	if(Simulation.isInMouseFunctionMode)
	{
		//glColor3d(1.0, 1.0, 1.0);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    		glEnable(GL_DEPTH_TEST);
		glColor4f(1.0, 1.0, 1.0, 1.0);
		glPushMatrix();
		glTranslatef(MouseX, MouseY, MouseZ);
		
		renderSphere(HitMultiplier*RadiusOfLeftAtrium,20,20);
		//renderSphere(5.0*NodeRadiusAdjustment*RadiusOfAtria,20,20);
		glPopMatrix();
		glDisable(GL_BLEND);
	}
	
	// Saves the picture if a movie is being recorded.
	if(Simulation.isRecording)
	{
		glReadPixels(5, 5, XWindowSize, YWindowSize, GL_RGBA, GL_UNSIGNED_BYTE, Buffer);
		fwrite(Buffer, sizeof(int)*XWindowSize*YWindowSize, 1, MovieFile);
	}

}


/* 
	 This function creates the GUI using ImGui.
	 This is where the actual window is built

	 All ImGui fields need to be in an if statement to check if the value has changed.
	 ImGui::CollapsingHeader to create a collapsible section
	 ImGui::Text to display text
	 ImGui::Input<Type> to create input fields for user input
	 ImGui::Slider<Type> to create sliders for user input
	 ImGui::Checkbox to create checkboxes for toggling options (must be bools)
	 ImGui::Combo to create dropdown menus for selecting options (must be int pointers)
	 ImGui::Button to create buttons for actions
	 ImGui::TextColored to display colored text (use vec4 to apply the color)
	 ImGui::SameLine to place elements on the same line
	 ImGui::isItemHovered to check if an item is hovered over (used for tooltips)

*/
void createGUI()
{
    // Setup ImGui window flags
    ImGuiWindowFlags window_flags = 0; // Initialize window flags to 0, flags are used to set window properties, like size, position, etc.
    window_flags |= ImGuiWindowFlags_AlwaysAutoResize; // Always resize the window to fit the content
    
    // Main Controls Window
    ImGui::Begin("Atrium Controls", NULL, window_flags); //title of the window, NULL means no pointer to a bool to close the window, window_flags are the flags we set above
    
    // Run/Pause button
    if (ImGui::Button(Simulation.isPaused ? "Run" : "Pause"))
    {
        Simulation.isPaused = !Simulation.isPaused;
    }
    
    // General simulation controls
    if (ImGui::CollapsingHeader("Simulation Controls", ImGuiTreeNodeFlags_DefaultOpen)) //open by default
    {

        // Contraction toggle (do we need this?? added it anyways)
        // bool contractionOn = Simulation.ContractionisOn;
        // if (ImGui::Checkbox("Contraction", &contractionOn)) 
        // {
        //     Simulation.ContractionisOn = contractionOn;
        // }
        


        // View controls
        bool frontHalf = Simulation.DrawFrontHalfFlag == 1; //Needed because ImGui needs a bool for a checkbox, can make a dropbox if more display options are needed
        if(ImGui::Checkbox("Draw Front Half Only", &frontHalf)) //checkbox for if we only want to draw the first half of the nodes
        {
            Simulation.DrawFrontHalfFlag = frontHalf ? 1 : 0;
            drawPicture();
        }
        
        // Node display options
        const char* nodeOptions[] = { "Off", "Half", "Full" }; //array of options for the dropdown menu
        int nodeDisplay = Simulation.DrawNodesFlag;

		//Combo makes a dropdown menu with the options in the array
        if(ImGui::Combo("Show Nodes", &nodeDisplay, nodeOptions, 3)) //args are menu name, pointer to the selected option, array of text options, # of options
        {
            if (nodeDisplay != Simulation.DrawNodesFlag) // Only update if the value changes
            {
                Simulation.DrawNodesFlag = nodeDisplay;
                drawPicture();
            }
        }
        
        // Projection mode {Do we REAAALLY need this??}
        // bool frustumView = Simulation.ViewFlag == 1;
        // if (ImGui::Checkbox("Frustum View", &frustumView))
        // {
        //     if (frustumView && Simulation.ViewFlag == 0)
        //     {
        //         Simulation.ViewFlag = 1;
        //         frustumView();
        //     }
        //     else if (!frustumView && Simulation.ViewFlag == 1)
        //     {
        //         Simulation.ViewFlag = 0;
        //         orthogonalView();
        //     }
        // }
        
        // Button for recording
		if (Simulation.isRecording)
		{
			if (ImGui::Button("Stop Recording"))
			{
				movieOff();
				Simulation.isRecording = false;
			}
		}
		else
		{
			if (ImGui::Button("Record Video"))
			{
				movieOn();
				Simulation.isRecording = true;
			}
		}

        // Screenshot button
        if (ImGui::Button("Screenshot"))
        {
            screenShot();
        }
    }
    
		// View angle controls
		if (ImGui::CollapsingHeader("View Controls", ImGuiTreeNodeFlags_DefaultOpen))//2nd arg is the flags, DefaultOpen means it will be open by default
		{
			// Predefined views
			if (ImGui::Button("PA"))
			{ 
				setView(4); 
				copyNodesToGPU(); 
				drawPicture(); 
			}
			if (ImGui::IsItemHovered())
			{
				ImGui::BeginTooltip();
				ImGui::Text("Posterior-Anterior View\nView from back to front");
				ImGui::EndTooltip();
			}
			
			ImGui::SameLine();
			if (ImGui::Button("AP"))  
			{
				setView(2); 
				copyNodesToGPU(); 
				drawPicture(); 
			}
			if (ImGui::IsItemHovered())
			{
				ImGui::BeginTooltip();
				ImGui::Text("Anterior-Posterior View\nView from front to back");
				ImGui::EndTooltip();
			}
			
			ImGui::SameLine();
			if (ImGui::Button("Ref"))
			{ 
				setView(6); 
				copyNodesToGPU(); 
				drawPicture(); 
			}
			if (ImGui::IsItemHovered())
			{
				ImGui::BeginTooltip();
				ImGui::Text("Reference View\nStandard orientation with pulmonary veins visible");
				ImGui::EndTooltip();
			}
			
			if (ImGui::Button("LAO"))
			{ 
				setView(1); 
				copyNodesToGPU(); 
				drawPicture(); 
			}
			if (ImGui::IsItemHovered())
			{
				ImGui::BeginTooltip();
				ImGui::Text("Left Anterior Oblique\nAngled view from front-left");
				ImGui::EndTooltip();
			}
			
			ImGui::SameLine();
			if (ImGui::Button("RAO"))
			{ 
				setView(3); 
				copyNodesToGPU(); 
				drawPicture(); 
			}
			if (ImGui::IsItemHovered())
			{
				ImGui::BeginTooltip();
				ImGui::Text("Right Anterior Oblique\nAngled view from front-right");
				ImGui::EndTooltip();
			}
			
			ImGui::SameLine();
			if (ImGui::Button("LL"))
			{ 
				setView(7); 
				copyNodesToGPU(); 
				drawPicture(); 
			}
			if (ImGui::IsItemHovered())
			{
				ImGui::BeginTooltip();
				ImGui::Text("Left Lateral\nDirect view from left side");
				ImGui::EndTooltip();
			}

			if (ImGui::Button("RL"))
			{ 
				setView(9); 
				copyNodesToGPU(); 
				drawPicture(); 
			}
			if (ImGui::IsItemHovered())
			{
				ImGui::BeginTooltip();
				ImGui::Text("Right Lateral\nDirect view from right side");
				ImGui::EndTooltip();
			}
			
			ImGui::SameLine();
			if (ImGui::Button("SUP"))
			{ 
				setView(8); 
				copyNodesToGPU(); 
				drawPicture(); 
			}
			if (ImGui::IsItemHovered())
			{
				ImGui::BeginTooltip();
				ImGui::Text("Superior View\nView from above (top-down)");
				ImGui::EndTooltip();
			}
			
			ImGui::SameLine();
			if (ImGui::Button("INF"))
			{ 
				setView(5); 
				copyNodesToGPU(); 
				drawPicture(); 
			}
			if (ImGui::IsItemHovered())
			{
				ImGui::BeginTooltip();
				ImGui::Text("Inferior View\nView from below (bottom-up)");
				ImGui::EndTooltip();
			}
		}
    
	// Mouse mode selection
	if (ImGui::CollapsingHeader("Mouse Functions", ImGuiTreeNodeFlags_DefaultOpen))
	{
		// Display current mouse mode
		ImGui::Text("Current Mode: ");
		if (!Simulation.isInMouseFunctionMode) 
		{
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, 1.0f), "Mouse Off");
		}
		else if (Simulation.isInAblateMode) 
		{
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Ablate Mode");
			ImGui::Text("Click on nodes to ablate");
			ImGui::Text("(Left Click: Ablate, Right Click: Undo)");
		}
		else if (Simulation.isInEctopicBeatMode) 
		{
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Ectopic Beat");
		} 
		else if (Simulation.isInEctopicEventMode) 
		{
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(0.0f, 0.5f, 1.0f, 1.0f), "Ectopic Trigger");
		} 
		else if (Simulation.isInAdjustMuscleAreaMode) 
		{
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Adjust Area");
		} 
		else if (Simulation.isInAdjustMuscleLineMode) 
		{
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Adjust Line");
		} 
		else if (Simulation.isInFindNodeMode) 
		{
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(0.5f, 0.0f, 1.0f, 1.0f), "Identify Node");
		}

		// Mouse mode buttons
		if (ImGui::Button("Mouse Off"))
		{
			mouseFunctionsOff();
			Simulation.isInMouseFunctionMode = false;
		}
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("Disables all mouse interaction with the model");
			ImGui::EndTooltip();
		}

		if (ImGui::Button("Ablate Mode")) 
		{
			mouseAblateMode();
			Simulation.isInMouseFunctionMode = true;
			Simulation.isInAblateMode = true;
		}
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("Left-click to ablate nodes\nRight-click to undo ablation");
			ImGui::EndTooltip();
		}

		if (ImGui::Button("Ectopic Beat")) 
		{
			mouseEctopicBeatMode();
			Simulation.isInMouseFunctionMode = true;
			Simulation.isInEctopicBeatMode = true;
		}
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("Left-click to set a node as an ectopic beat node\nwith a constant beat period");
			ImGui::EndTooltip();
		}

		if (ImGui::Button("Ectopic Trigger")) 
		{
			mouseEctopicEventMode();
			Simulation.isInMouseFunctionMode = true;
			Simulation.isInEctopicEventMode = true;
		}
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("Left-click to trigger a single pulse at a node");
			ImGui::EndTooltip();
		}

		if (ImGui::Button("Adjust Area"))
		{
			mouseAdjustMusclesAreaMode();
			Simulation.isInMouseFunctionMode = true;
			Simulation.isInAdjustMuscleAreaMode = true;
		}
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("Left-click to adjust muscle properties in an area\nAffects refractory period and conduction velocity\n\nRight-click to undo adjustment");
			ImGui::EndTooltip();
		}

		if (ImGui::Button("Adjust Line")) 
		{
			mouseAdjustMusclesLineMode();
			Simulation.isInMouseFunctionMode = true;
			Simulation.isInAdjustMuscleLineMode = true;
		}
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("Left-click to adjust muscle properties along a line\nAffects refractory period and conduction velocity\n\nRight-click to undo adjustment");
			ImGui::EndTooltip();
		}

		if (ImGui::Button("Identify Node")) 
		{
			mouseIdentifyNodeMode();
			Simulation.isInMouseFunctionMode = true;
			Simulation.isInFindNodeMode = true;
		}
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("Left-click to display the ID of a node");
			ImGui::EndTooltip();
		}

		// Display identified nodes
		if (Simulation.isInFindNodeMode) 
		{
			ImGui::TextColored(ImVec4(1.0f, 0.5f, 1.0f, 1.0f), "Click on nodes to identify them");
			
			// Display any identified nodes (purple nodes)
			ImGui::BeginChild("IdentifiedNodes", ImVec2(0, 120), true); // Create a child window for displaying identified nodes, imVec2(0, 120) sets the size of the child window (0 means auto width, 120 is height), true means bordered
			bool foundAny = false; // Flag to check if any nodes are identified
			
			for (int i = 0; i < NumberOfNodes; i++)
			{
				// Check if node is marked as drawn and is the purple identify color
				if (Node[i].isDrawNode && 
					Node[i].color.x == 1.0f && 
					Node[i].color.y == 0.0f && 
					Node[i].color.z == 1.0f)
				{
					foundAny = true;
					ImGui::Text("Node ID: %d", i);
				}
			}
			
			if (!foundAny) {
				ImGui::TextDisabled("No nodes identified yet");
			}
			
			ImGui::EndChild();
			
			if (ImGui::Button("Clear Identified Nodes"))
			{
				// Reset all purple nodes back to default
				for (int i = 0; i < NumberOfNodes; i++)
				{
					if (Node[i].isDrawNode && Node[i].color.x == 1.0f && Node[i].color.y == 0.0f && Node[i].color.z == 1.0f) //if the node is purple and is drawn
					{
						if (Node[i].isAblated) 
						{
							// Reset ablated nodes to ablated color (white)
							Node[i].color.x = 1.0f;
							Node[i].color.y = 1.0f; 
							Node[i].color.z = 1.0f;
							// isDrawNode stays true for ablated nodes
						}
						else 
						{
							// Reset non-ablated nodes to default color (green)
							Node[i].isDrawNode = false;
							Node[i].color.x = 0.0f;
							Node[i].color.y = 1.0f; 
							Node[i].color.z = 0.0f;
						}
					}
				}
				
				// Update the GPU with changes
				copyNodesMusclesToGPU();
				drawPicture();
			}
		}

		// Selection area slider
		float hitMult = HitMultiplier;
		if (ImGui::SliderFloat("Selection Area", &hitMult, 0.0f, 0.2f, "%.3f", ImGuiSliderFlags_AlwaysClamp))
		{
			HitMultiplier = hitMult;
		}
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("Adjusts the size of the selection area\nLarger values effect more nodes");
			ImGui::EndTooltip();
		}

		//Muscle adjustment sliders
		if (Simulation.isInAdjustMuscleAreaMode || Simulation.isInAdjustMuscleLineMode)
		{
			ImGui::Separator(); //add a line to separate the sections
			ImGui::Text("Muscle Adjustment Parameters");
			ImGui::NewLine(); //add a new line for spacing
			

			ImGui::Text("Refractory Period Multiplier");
			float refractoryMultiplier = RefractoryPeriodAdjustmentMultiplier;
			ImGui::SetNextItemWidth(150); // Narrower slider to make room for input
			if (ImGui::SliderFloat("##refractoryMultiplier", &refractoryMultiplier, 0.1f, 5.0f, "%.2f")) 
			{
				RefractoryPeriodAdjustmentMultiplier = refractoryMultiplier;
			}
			if (ImGui::IsItemHovered())
			{
				ImGui::BeginTooltip();
				ImGui::Text("Adjusts the refractory period of the muscle\n\nDrag the slider or enter a precise value in the input box");
				ImGui::EndTooltip();
			}

			ImGui::SameLine();
			ImGui::SetNextItemWidth(60); // Fixed width for input box
			float originalRefMultiplier = refractoryMultiplier;
			if (ImGui::InputFloat("##refractoryInput", &refractoryMultiplier, 0, 0, "%.2f"))
			{
				// Clamp to valid range
				refractoryMultiplier = (refractoryMultiplier < 0.1f) ? 0.1f : (refractoryMultiplier > 5.0f ? 5.0f : refractoryMultiplier);
				
				// Update if changed
				if (refractoryMultiplier != originalRefMultiplier)
				{
					RefractoryPeriodAdjustmentMultiplier = refractoryMultiplier;
				}
			}


			ImGui::SameLine();
			if (ImGui::Button("Reset##1")) 
			{
				RefractoryPeriodAdjustmentMultiplier = 1.0f;
			}

			// For the Conduction Velocity Multiplier slider:
			ImGui::Text("Conduction Velocity Multiplier");
			float conductionMultiplier = MuscleConductionVelocityAdjustmentMultiplier;
			ImGui::SetNextItemWidth(150); // Narrower slider to make room for input
			if (ImGui::SliderFloat("##conductionVelocityMultiplier", &conductionMultiplier, 0.1f, 5.0f, "%.2f")) 
			{
				MuscleConductionVelocityAdjustmentMultiplier = conductionMultiplier;
			}
			if (ImGui::IsItemHovered())
			{
				ImGui::BeginTooltip();
				ImGui::Text("Adjusts the refractory period of the muscle\n\nDrag the slider or enter a precise value in the input box");
				ImGui::EndTooltip();
			}

			ImGui::SameLine();
			ImGui::SetNextItemWidth(60); // Fixed width for input box
			float originalConductionMultiplier = conductionMultiplier;
			if (ImGui::InputFloat("##conductionInput", &conductionMultiplier, 0, 0, "%.2f"))
			{
				// Clamp to valid range
				conductionMultiplier = (conductionMultiplier < 0.1f) ? 0.1f : (conductionMultiplier > 5.0f ? 5.0f : conductionMultiplier);
				
				// Update if changed
				if (conductionMultiplier != originalConductionMultiplier)
				{
					MuscleConductionVelocityAdjustmentMultiplier = conductionMultiplier;
				}
			}

			ImGui::SameLine();
			if (ImGui::Button("Reset##2"))
			{
				MuscleConductionVelocityAdjustmentMultiplier = 1.0f;
			}
		}
	}
    
    // Heartbeat controls
    if (ImGui::CollapsingHeader("Heartbeat Controls"))
    {
		//Slider for beat period of the Pulse Node
		ImGui::Text("Beat Period (ms)");
        float beatPeriod = Node[PulsePointNode].beatPeriod;
		float beatPeriodMin = 10.0f; // Minimum value for beat period
		float beatPeriodMax = 1000.0f; // Maximum value for beat period

        if (ImGui::SliderFloat("##beatPeriodSlider", &beatPeriod, beatPeriodMin, beatPeriodMax, "%.1f ms")) 
		{
            Node[PulsePointNode].beatPeriod = beatPeriod;
            cudaMemcpy(NodeGPU, Node, NumberOfNodes*sizeof(nodeAttributesStructure), cudaMemcpyHostToDevice);
            cudaErrorCheck(__FILE__, __LINE__);
		}
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("Adjust period of time between beats from the pulse node\n\nDrag the slider or enter a precise value in the input box");
			ImGui::EndTooltip();
		}



		ImGui::SameLine();

		//Input field for beat period of the Pulse Node
		ImGui::SetNextItemWidth(60);  // Make the input field smaller, fixed 60 pixels
		
		float originalBeatPeriod = beatPeriod; //Store the original value to check if it changed
		if (ImGui::InputFloat("##beatPeriodInput", &beatPeriod, 0, 0, "%.1f")) 
		{
			//make sure input is valid
			beatPeriod = (beatPeriod < beatPeriodMin) ? beatPeriodMin : (beatPeriod >  beatPeriodMax ?  beatPeriodMax : beatPeriod);
			
			// If value actually changed, update
			if (beatPeriod != originalBeatPeriod) 
			{
				Node[PulsePointNode].beatPeriod = beatPeriod;
				cudaMemcpy(NodeGPU, Node, NumberOfNodes*sizeof(nodeAttributesStructure), cudaMemcpyHostToDevice);
				cudaErrorCheck(__FILE__, __LINE__);
			}
		}
        
        if (ImGui::Button("+ 10ms")) 
		{
            Node[PulsePointNode].beatPeriod += 10;
            cudaMemcpy(NodeGPU, Node, NumberOfNodes*sizeof(nodeAttributesStructure), cudaMemcpyHostToDevice);
            cudaErrorCheck(__FILE__, __LINE__);
        }

        ImGui::SameLine();
        if (ImGui::Button("- 10ms")) 
		{
            Node[PulsePointNode].beatPeriod -= 10;
            if(Node[PulsePointNode].beatPeriod < 0) // Prevent negative beat period 
			{
                Node[PulsePointNode].beatPeriod = 0;
            }
            cudaMemcpy(NodeGPU, Node, NumberOfNodes*sizeof(nodeAttributesStructure), cudaMemcpyHostToDevice);
            cudaErrorCheck(__FILE__, __LINE__);
        }
        
        //Ectopic beat Sliders
        ImGui::Separator();
        ImGui::Text("Ectopic Beats");
        
        // Show sliders for each ectopic beat node
        bool hasEctopicBeats = false; // flag to see if we have any ectopic beats; consider adding to simulation struct?
        for(int i = 0; i < NumberOfNodes; i++) 
		{
            if(Node[i].isBeatNode && i != PulsePointNode) //if this is an ectopic beat node and not the pulse node
			{
                hasEctopicBeats = true;
                
                char nodeName[32];
                sprintf(nodeName, "Ectopic Beat Node %d", i);
                
                if (ImGui::TreeNode(nodeName))  //a tree node is a collapsible section, so we can have multiple ectopic beats in the same window
				{
					ImGui::NewLine(); //add a new line for spacing

					ImGui::Text("Ectopic Beat Period (ms)");
					float beatPeriod = Node[i].beatPeriod;

					ImGui::SetNextItemWidth(150); // Narrower slider so the input box fits better

					if (ImGui::SliderFloat("##EctopicBeatPeriod", &beatPeriod, 10.0f, 1000.0f, "%.1f ms")) 
					{
						Node[i].beatPeriod = beatPeriod;
						cudaMemcpy(NodeGPU, Node, NumberOfNodes*sizeof(nodeAttributesStructure), cudaMemcpyHostToDevice);
						cudaErrorCheck(__FILE__, __LINE__);
					}
					if (ImGui::IsItemHovered())
					{
						ImGui::BeginTooltip();
						ImGui::Text("Controls how often this node beats\n\nDrag the slider or enter a precise value in the input box");
						ImGui::EndTooltip();
					}

					ImGui::SameLine();
					ImGui::SetNextItemWidth(60);
					float originalBeatPeriod = beatPeriod;
					if (ImGui::InputFloat("##beatPeriodInput", &beatPeriod, 0, 0, "%.1f"))
					{
						// Clamp to valid range
						beatPeriod = (beatPeriod < 10.0f) ? 10.0f : (beatPeriod > 1000.0f ? 1000.0f : beatPeriod);
						
						// Update if changed
						if (beatPeriod != originalBeatPeriod)
						{
							Node[i].beatPeriod = beatPeriod;
							cudaMemcpy(NodeGPU, Node, NumberOfNodes*sizeof(nodeAttributesStructure), cudaMemcpyHostToDevice);
							cudaErrorCheck(__FILE__, __LINE__);
						}
					}

					// Delay/time until next beat slider for ectopic beats
					ImGui::Text("Time Until Next Beat (ms)");
					float timeDelay = Node[i].beatPeriod - Node[i].beatTimer;
					ImGui::SetNextItemWidth(150); // Narrower slider
					if (ImGui::SliderFloat("##ectopicBeatPeriodDelay", &timeDelay, 0.0f, Node[i].beatPeriod, "%.1f ms")) 
					{
						// Convert back to beatTimer when storing
						Node[i].beatTimer = Node[i].beatPeriod - timeDelay;
						cudaMemcpy(NodeGPU, Node, NumberOfNodes*sizeof(nodeAttributesStructure), cudaMemcpyHostToDevice);
						cudaErrorCheck(__FILE__, __LINE__);
					}
					if (ImGui::IsItemHovered())
					{
						ImGui::BeginTooltip();
						ImGui::Text("Controls how long until this node beats\n\nDrag the slider or enter a precise value in the input box");
						ImGui::EndTooltip();
					}

					ImGui::SameLine();
					ImGui::SetNextItemWidth(60);
					float originalTimeDelay = timeDelay;
					if (ImGui::InputFloat("##timeDelayInput", &timeDelay, 0, 0, "%.1f"))
					{
						// Clamp to valid range
						timeDelay = (timeDelay < 0.0f) ? 0.0f : (timeDelay > Node[i].beatPeriod ? Node[i].beatPeriod : timeDelay);
						
						// Update if changed
						if (timeDelay != originalTimeDelay)
						{
							Node[i].beatTimer = Node[i].beatPeriod - timeDelay;
							cudaMemcpy(NodeGPU, Node, NumberOfNodes*sizeof(nodeAttributesStructure), cudaMemcpyHostToDevice);
							cudaErrorCheck(__FILE__, __LINE__);
						}
					}
                    
					//button to remove ectopic beat nodes
                    if (ImGui::Button("Delete Ectopic Beat")) 
					{
                        Node[i].isBeatNode = false;
                        Node[i].isDrawNode = false;
                        Node[i].color = {0.0f, 1.0f, 0.0f, 1.0f}; // Reset color
                        cudaMemcpy(NodeGPU, Node, NumberOfNodes*sizeof(nodeAttributesStructure), cudaMemcpyHostToDevice);
                        cudaErrorCheck(__FILE__, __LINE__);
                    }
                    
                    ImGui::TreePop(); // Close the tree node
                }
            }
        }
        
        if (!hasEctopicBeats) //if there are no ectopic beats, show a message
		{
            ImGui::TextDisabled("No ectopic beats configured."); //TextDisabled makes it greyed out
            ImGui::Text("Use the Ectopic Beat button to add one.");
        }
    }
    
    // Utility functions
    if (ImGui::CollapsingHeader("Utilities"))
    {
        if (ImGui::Button("Save Settings"))
		{
            saveSettings();
        }
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("Save current muscle properties and simulation\nsettings to a file for later use");
			ImGui::EndTooltip();
		}
        
        if (ImGui::Button("Find Nodes"))
		{
            copyNodesMusclesFromGPU();

			// Reset previous nodes properly
			if (Simulation.frontNodeIndex >= 0 && Simulation.topNodeIndex >= 0) //if the front and top node indices are valid
			
			{
				// Reset front node based on ablation status
				if (Node[Simulation.frontNodeIndex].isAblated) 
				{
					Node[Simulation.frontNodeIndex].isDrawNode = true; // Keep it visible, set to white
					Node[Simulation.frontNodeIndex].color.x = 1.0f;
					Node[Simulation.frontNodeIndex].color.y = 1.0f;
					Node[Simulation.frontNodeIndex].color.z = 1.0f;
				} 
				else 
				{
					Node[Simulation.frontNodeIndex].isDrawNode = false; //back to default color
					Node[Simulation.frontNodeIndex].color.x = 0.0f;
					Node[Simulation.frontNodeIndex].color.y = 1.0f;
					Node[Simulation.frontNodeIndex].color.z = 0.0f;
				}

				// Reset top node based on ablation status
				if (Node[Simulation.topNodeIndex].isAblated) 
				{
					Node[Simulation.topNodeIndex].isDrawNode = true; // Keep it visible, set color to white
					Node[Simulation.topNodeIndex].color.x = 1.0f;
					Node[Simulation.topNodeIndex].color.y = 1.0f;
					Node[Simulation.topNodeIndex].color.z = 1.0f;
				} 
				else 
				{
					Node[Simulation.topNodeIndex].isDrawNode = false; //back to default color
					Node[Simulation.topNodeIndex].color.x = 0.0f;
					Node[Simulation.topNodeIndex].color.y = 1.0f;
					Node[Simulation.topNodeIndex].color.z = 0.0f;
				}
			}

			//give bad values to the indices so we know they are not valid unless they are made valid again
            float maxZ = -10000.0;
            float maxY = -10000.0;
            int indexZ = -1;
            int indexY = -1;
            
            for(int i = 0; i < NumberOfNodes; i++) // Loop through all nodes, checking for the max Z and Y values
			{
                if(maxZ < Node[i].position.z) 
				{
                    maxZ = Node[i].position.z;
                    indexZ = i;
                }
                
                if(maxY < Node[i].position.y) 
				{
                    maxY = Node[i].position.y;
                    indexY = i;
                }
            }
            
			//set the colors of the nodes to blue and purple, respectively
			Node[indexZ].isDrawNode = true; // Set the front node to be drawn as blue
            Node[indexZ].color.x = 0.0;
            Node[indexZ].color.y = 0.0;
            Node[indexZ].color.z = 1.0;
            
			Node[indexY].isDrawNode = true; // Set the top node to be drawn as purple
            Node[indexY].color.x = 1.0;
            Node[indexY].color.y = 0.0;
            Node[indexY].color.z = 1.0;
            
			// Store indices for persistent display
			Simulation.frontNodeIndex = indexZ;
			Simulation.topNodeIndex = indexY;
			Simulation.nodesFound = true;
            
            drawPicture(); // Redraw the picture to show the new colors
            copyNodesMusclesToGPU(); // Copy the updated nodes back to GPU (since the color changed)
        }
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("Identify and highlight the front (blue) and top (purple)\nnodes in the current view orientation\n\nIt is reccomended to draw nodes to see the results clearly");
			ImGui::EndTooltip();
		}

		// Display the information outside the button handler so it persists
		if (Simulation.nodesFound) 
		{
			ImGui::Separator();
			ImGui::Text("Front node (blue): %d", Simulation.frontNodeIndex);
			ImGui::Text("Top node (purple): %d", Simulation.topNodeIndex);
		}
    }

	//Display movement controls
	if (ImGui::CollapsingHeader("Keyboard Controls"))
	{

		ImGui::Text("Quit: q");
		ImGui::NewLine(); //add a new line for spacing
		ImGui::Text("Rotate X-axis: a/d");
		ImGui::Text("Rotate Y-axis: w/s");
		ImGui::Text("Rotate Z-axis: z/Z");
		ImGui::Text("Zoom In/Out: e/E");
		
	}
    
    ImGui::End(); //end the main controls window
    
    // Beginning of stats window
	//if there's any relevant information we should show for quick viewing, put it here., we can add toggles for what to show in the main window if we want to.

    ImGui::Begin("Simulation Stats", NULL, window_flags);
	
    ImGui::Text("Run time: %.2f ms", RunTime);
    ImGui::Text("Beat rate: %.2f ms", Node[PulsePointNode].beatPeriod);
    
    if(Simulation.isInAdjustMuscleAreaMode || Simulation.isInAdjustMuscleLineMode) 
	{
        ImGui::Separator();
        ImGui::Text("Refractory multiplier: %.3f", RefractoryPeriodAdjustmentMultiplier);
        ImGui::Text("Conduction multiplier: %.3f", MuscleConductionVelocityAdjustmentMultiplier);
    }
    
	// Print ectopic beat nodes and their periods
    ImGui::Separator();
    for(int i = 0; i < NumberOfNodes; i++) 
	{
        if(Node[i].isBeatNode && i != PulsePointNode) 
		{
            ImGui::Text("Ectopic Beat Node %d: %.2f ms", i, Node[i].beatPeriod);
        }
    }
  
    ImGui::End(); //end of stats window
}