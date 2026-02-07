/* This file contains:
 1: All the functions that determine how to orient and view the simulation.
 2: All the functions that draw the actual simulation. 
 3: The functions that print to the linux terminal all the setting of the simulation.
 In short this file holds the functions that present information to the user.

 floa4 findCenterOfMass();
 void renderSphere();
 void sphereVBO();
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
		printf("\n The mass is too small.");
		printf("\n The simulation has been terminated.\n\n");
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
	NumSphereVertices = vertices.size() / 6; // 6 floats per vertex (pos + normal)
	NumSphereIndices = indices.size();

	// Create and setup OpenGL buffers on the GPU
	// - Generate unique buffer IDs
	// - Bind buffers to set them as active
	// - Copy data from CPU arrays to GPU memory
	glGenBuffers(1, &SphereVBO);  // Generate Vertex Buffer Object for storing positions and normals
	glBindBuffer(GL_ARRAY_BUFFER, SphereVBO);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

	// Same process for the index buffer
	glGenBuffers(1, &SphereIBO);  // Generate Index Buffer Object for storing triangle connections
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, SphereIBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

	// Unbind buffers to prevent accidental modification
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void renderSphereVBO() 
{
    // Bind the VBO and IBO
    glBindBuffer(GL_ARRAY_BUFFER, SphereVBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, SphereIBO);
    
    // Enable vertex and normal arrays
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    
    // Set up pointers to vertex and normal data
    glVertexPointer(3, GL_FLOAT, 6 * sizeof(float), 0);
    glNormalPointer(GL_FLOAT, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    
    // Draw the sphere
    glDrawElements(GL_TRIANGLES, NumSphereIndices, GL_UNSIGNED_INT, 0);
    
    // Disable arrays
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    
    // Unbind buffers
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}


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
 This function centers the LA and resets the center of view to (0, 0, 0).
 It is called periodically in a running simulation to center the LA, because the LA is not symmetrical 
 and will wander off over time. It is also use to center the LA before all the views are set.
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
	
	//if(!Simulation.isPaused) glColor3d(0.0,1.0,0.0); // Green is running
	//else glColor3d(1.0,0.0,0.0); // Red is paused	
	glColor3d(Node[PulsePointNode].color.x, Node[PulsePointNode].color.y, Node[PulsePointNode].color.z);
	glPushMatrix();
	glTranslatef(Node[PulsePointNode].position.x, Node[PulsePointNode].position.y, Node[PulsePointNode].position.z);
	renderSphereVBO();
	glPopMatrix();
	
	// Drawing center node
	//This draws a node at the center of the simulation for debugging purposes
	if(false) // false turns it off, true turns it on.
	{
		glColor3d(0.0,0.0,1.0);
		glPushMatrix();
		glTranslatef(CenterOfSimulation.x, CenterOfSimulation.y, CenterOfSimulation.z);
		renderSphereVBO();
		glPopMatrix();
	}
	
	// Drawing nodes
	if(Simulation.DrawNodesFlag == 1 || Simulation.DrawNodesFlag == 2)  //if we're drawing half(1) or all(2) of the nodes
	{
		for(int i = 0; i < NumberOfNodes; i++) // Start at 1 to skip the pulse node and go through all nodes
		{
			if(Simulation.DrawFrontHalfFlag == 1 || Simulation.DrawNodesFlag == 1) //If we're only drawing the nodes on the front half.
			{
				if(CenterOfSimulation.z - 0.001 < Node[i].position.z)  //Draw only the nodes in the front half.
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
	// If the nodes are not drawn as spheres you will be in this else case.
	// Now some of the node we still draw as points, like node that have been ablated, ectopic nodes, or nodes connected to muscle
	// that have been adjusted. This helps the user keep track of what has been done. This is what is done here and is based on
	// the .isDrawNode flag.
	else 
	{
		glPointSize(NodePointSize);
		glBegin(GL_POINTS);
	 	for(int i = 0; i < NumberOfNodes; i++)
		{
			if(Simulation.DrawFrontHalfFlag == 1)
			{
				if(CenterOfSimulation.z - 0.001 < Node[i].position.z)  // Only drawing the nodes in the front half.
				{
					glColor3d(Node[i].color.x, Node[i].color.y, Node[i].color.z);
					/*if(Node[i].isDrawNode)
					{
						glVertex3f(Node[i].position.x, Node[i].position.y, Node[i].position.z);
					}*/
				}
			}
			else
			{
				glColor3d(Node[i].color.x, Node[i].color.y, Node[i].color.z);
				/*if(Node[i].isDrawNode)
				{
					glVertex3f(Node[i].position.x, Node[i].position.y, Node[i].position.z);
				}*/
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
					if(CenterOfSimulation.z - 0.001 < Node[i].position.z && CenterOfSimulation.z - 0.001 < Node[k].position.z)  // Only drawing the nodes in the front half.
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

	 For buttons and checkboxes, its best to use ternary operators when posssible
*/
void createGUI()
{

	// Get actual viewport size -- this is the size of the window, not the size of the the openGL viewport
	const ImGuiViewport* viewport = ImGui::GetMainViewport();

	//Set in top right corner of the window, 10px offset from both edges
	//ImGUICond_Always means the position will always be set to this value, regardless of previous positions
	//last arg anchors to the right and top of the window
	ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - 10, viewport->WorkPos.y + 10), ImGuiCond_Always,  ImVec2(1.0f, 0.0f));

    // Setup ImGui window flags
    ImGuiWindowFlags window_flags = 0; // Initialize window flags to 0, flags are used to set window properties, like size, position, etc. 0 means no flags are set
    window_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing; // Always resize the window to fit the content
    

	// Set the collapsed state if guiCollapsed is true (toggled by ctrl + h callback)
    ImGui::SetNextWindowCollapsed(Simulation.guiCollapsed, ImGuiCond_Always);


    // Main Controls Window
    ImGui::Begin("Control Panel", NULL, window_flags); //title of the window, NULL means no pointer to a bool to close the window, window_flags are the flags we set above
    
	//update bool to match current state (makes sure clicking also works in addition to ctrl + h)
	Simulation.guiCollapsed = ImGui::IsWindowCollapsed();


    
    // General simulation controls
    if (ImGui::CollapsingHeader("Simulation Controls", ImGuiTreeNodeFlags_DefaultOpen)) //open by default
    {
        // View controls
        bool frontHalf = Simulation.DrawFrontHalfFlag == 1; //Needed because ImGui needs a bool for a checkbox, can make a dropbox if more display options are needed
        if(ImGui::Checkbox("Draw Front Half Only", &frontHalf)) //checkbox for if we only want to draw the first half of the nodes
        {
			//when the button is pressed it will change the value of frontHalf to the opposite of what it was before
            Simulation.DrawFrontHalfFlag = frontHalf ? 1 : 0;
            drawPicture();
        }
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("(F2)");
			ImGui::EndTooltip();
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
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("(F3)");
			ImGui::EndTooltip();
		}
        
        // Change view
	

	}       
       
	// View presets
	if (ImGui::CollapsingHeader("View Controls", ImGuiTreeNodeFlags_DefaultOpen))//2nd arg is the flags, DefaultOpen means it will be open by default
	{

		if (ImGui::Button("PA"))
		{ 
			setView(4); 
			//copyNodesToGPU(); 
			drawPicture(); 
		}
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("(7)\nPosterior-Anterior View\nView from back to front");
			ImGui::EndTooltip();
		}
		
		ImGui::SameLine();
		if (ImGui::Button("AP"))  
		{
			setView(2); 
			//copyNodesToGPU(); 
			drawPicture(); 
		}
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("(8)\nAnterior-Posterior View\nView from front to back");
			ImGui::EndTooltip();
		}
		
		ImGui::SameLine();
		if (ImGui::Button("Ref"))
		{ 
			setView(6); 
			//copyNodesToGPU(); 
			drawPicture(); 
		}
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("(9)\nReference View\nStandard orientation with pulmonary veins visible");
			ImGui::EndTooltip();
		}
		
		if (ImGui::Button("LAO"))
		{ 
			setView(1); 
			//copyNodesToGPU(); 
			drawPicture(); 
		}
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("(4)\nLeft Anterior Oblique\nAngled view from front-left");
			ImGui::EndTooltip();
		}
		
		ImGui::SameLine();
		if (ImGui::Button("RAO"))
		{ 
			setView(3); 
			//copyNodesToGPU(); 
			drawPicture(); 
		}
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("(5)\nRight Anterior Oblique\nAngled view from front-right");
			ImGui::EndTooltip();
		}
		
		ImGui::SameLine();
		if (ImGui::Button("LL"))
		{ 
			setView(7); 
			//copyNodesToGPU(); 
			drawPicture(); 
		}
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("(6)\nLeft Lateral\nDirect view from left side");
			ImGui::EndTooltip();
		}

		if (ImGui::Button("RL"))
		{ 
			setView(9); 
			//copyNodesToGPU(); 
			drawPicture(); 
		}
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("(1)\nRight Lateral\nDirect view from right side");
			ImGui::EndTooltip();
		}
		
		ImGui::SameLine();
		if (ImGui::Button("SUP"))
		{ 
			setView(8); 
			//copyNodesToGPU(); 
			drawPicture(); 
		}
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("(2)\nSuperior View\nView from above (top-down)");
			ImGui::EndTooltip();
		}
		
		ImGui::SameLine();
		if (ImGui::Button("INF"))
		{ 
			setView(5); 
			//copyNodesToGPU(); 
			drawPicture(); 
		}
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("(3)\nInferior View\nView from below (bottom-up)");
			ImGui::EndTooltip();
		}
	}
	
	// Mouse mode selection
	if (ImGui::CollapsingHeader("Mouse Functions", ImGuiTreeNodeFlags_DefaultOpen))
	{
		// TODO: Rewrite this for the new mode system
		/* // Display current mouse mode
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
			ImGui::Text("Alt + Q to exit mouse mode");
			ImGui::Text("(Left Click: Ablate, Right Click: Undo)");
		}
		else if (Simulation.isInEctopicBeatMode) 
		{
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Ectopic Beat");
			ImGui::Text("Alt + Q to exit mouse mode");
		} 
		else if (Simulation.isInEctopicEventMode) 
		{
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(0.0f, 0.5f, 1.0f, 1.0f), "Ectopic Trigger");
			ImGui::Text("Alt + Q to exit mouse mode");
		} 
		else if (Simulation.isInAdjustMuscleAreaMode) 
		{
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Adjust Area");
			ImGui::Text("Alt + Q to exit mouse mode");
		} 
		else if (Simulation.isInAdjustMuscleLineMode) 
		{
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Adjust Line");
			ImGui::Text("Alt + Q to exit mouse mode");
		} 
		else if (Simulation.isInFindNodeMode) 
		{
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(0.5f, 0.0f, 1.0f, 1.0f), "Identify Node");
			ImGui::Text("Alt + Q to exit mouse mode");
		}*/

		// Mouse mode buttons
		if (ImGui::Button("Mouse Off"))
		{
			setMouseMode(-1); // OR whatever number for simulation off 
		}
		if (ImGui::IsItemHovered())
		{
			// TODO: add helper function
		}

		if (ImGui::Button("Set Normal Mode")) 
		{
			setMouseMode(0); // TODO: Determine numbers for different modes and replace with constants or an enum for readability
		}
		if (ImGui::IsItemHovered())
		{ // TODO: Make a helper function for tooltips since this is repeated so much
			ImGui::BeginTooltip();
			ImGui::Text("(F7)\nLeft-click to ablate nodes\nRight-click to undo ablation");
			ImGui::EndTooltip();
		}

		if (ImGui::Button("Set Bachmann's Bundle")) 
		{
			setMouseMode(1); // TODO: Determine numbers for different modes and replace with constants or an enum for readability
		}
		if(ImGui::IsItemHovered) 
		{ // TODO: Write Helper Funcitons
			// displayTooltip("Bachmann's Bundle Mode", "Left-click to select nodes for Bachmann's Bundle\nRight-click to undo selection");
		}
		if (ImGui::Button("Set Appendage")) 
		{
			setMouseMode(2); // TODO: Determine numbers for different modes and replace with constants or an enum for readability
		}
		if(ImGui::IsItemHovered) 
		{ // TODO: Write Helper Funcitons
			// displayTooltip("Bachmann's Bundle Mode", "Left-click to select nodes for Bachmann's Bundle\nRight-click to undo selection");
		}
		if (ImGui::Button("Scar Tissue")) 
		{
			// setMouseMode(MOUSE_MODE_SCAR_TISSUE); // TODO: add defines for mouse mode
		}
		if(ImGui::IsItemHovered) 
		{ // TODO: Write Helper Funcitons
			// displayTooltip("Bachmann's Bundle Mode", "Left-click to select nodes for Bachmann's Bundle\nRight-click to undo selection");
		}

	}
    
    // Utility functions
    if (ImGui::CollapsingHeader("Utilities"))
    {
		//Save settings button
        if (ImGui::Button("Save Settings"))
		{
            saveSettings();
        }
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("(F6)\nSave current muscle properties and simulation\nsettings to a file for later use");
			ImGui::EndTooltip();
		}

       

    }

	//Display movement controls
	if (ImGui::CollapsingHeader("Keyboard Controls"))
	{

		ImGui::Text("Quit: esc");
		ImGui::NewLine(); //add a new line for spacing
		ImGui::Text("Rotate X-axis: a/d; Left/Right");
		ImGui::Text("Rotate Y-axis: w/s; Up/Down");
		ImGui::Text("Rotate Z-axis: z/Z; Shift + Left/Right");
		ImGui::Text("Zoom In/Out: e/E; Shift + Up/Down");

		ImGui::Text("Collapse/Expand GUI: Ctrl + h");
		
	}
    
    ImGui::End(); //end the main controls window
    
  
}
