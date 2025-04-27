/*
 This file contains all the callBack functions and functions that it calls to do its work.
 This file contains all the ways a user can interact (Mouse and Terminal) with 
 a running simulation.
 
 The functions in this file are listed below and in this order.
 
 void reshape(GLFWwindow* window, int width, int height);
 void mouseFunctionsOff();
 void mouseAblateMode();
 void mouseEctopicBeatMode();
 void mouseAdjustMusclesAreaMode();
 void mouseAdjustMusclesLineMode();
 void mouseIdentifyNodeMode();
 bool setMouseMuscleAttributes();
 void setEctopicBeat(int nodeId);
 void clearStdin();
 string getTimeStamp();
 void movieOn();
 void movieOff();
 void screenShot();
 void saveSettings();
 void KeyPressed(GLFWwindow* window, int key, int scancode, int action, int mods);
 void mousePassiveMotionCallback(GLFWwindow* window, double x, double y);
 void myMouse(GLFWwindow* window, int button, int state, double x, double y);
 void scrollWheel(GLFWwindow*, double, double);

/*
 OpenGL callback when the window is reshaped.
*/
void reshape(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

/*
 Turns off all the user interactions.
*/
void mouseFunctionsOff()
{
	//Simulation.isPaused = true;
	Simulation.isInAblateMode = false;
	Simulation.isInEctopicBeatMode = false;
	Simulation.isInEctopicEventMode = false;
	Simulation.isInAdjustMuscleAreaMode = false;
	Simulation.isInAdjustMuscleLineMode = false;
	Simulation.isInFindNodeMode = false;
	Simulation.isInMouseFunctionMode = false;
	glfwSetCursor(Window, glfwCreateStandardCursor(GLFW_ARROW_CURSOR)); // Set cursor to default arrow.
	drawPicture();
}

/*
 Puts the user in ablate mode.
*/
void mouseAblateMode()
{
	mouseFunctionsOff();
	Simulation.isPaused = true;
	Simulation.isInAblateMode = true;
	Simulation.isInMouseFunctionMode = true;
	glfwSetInputMode(Window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN); // Set cursor to hidden.
	//orthogonalView();
	drawPicture();
}

/*
 Puts the user in ectopic beat mode.
*/
void mouseEctopicBeatMode()
{
	mouseFunctionsOff();
	Simulation.isPaused = true;
	Simulation.isInEctopicBeatMode = true;
	Simulation.isInMouseFunctionMode = true;
	glfwSetInputMode(Window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN); // Set cursor to hidden.
	//orthogonalView();
	drawPicture();
}

/*
 Puts the user in ectopic event mode.
*/
void mouseEctopicEventMode()
{
	mouseFunctionsOff();
	Simulation.isPaused = true;
	Simulation.isInEctopicEventMode = true;
	Simulation.isInMouseFunctionMode = true;
	glfwSetInputMode(Window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN); // Set cursor to hidden.
	//orthogonalView();
	drawPicture();
}

/*
 Puts the user in area muscle adjustment mode.
*/
void mouseAdjustMusclesAreaMode()
{
	mouseFunctionsOff();
	Simulation.isPaused = true;
	Simulation.isInAdjustMuscleAreaMode = true;
	Simulation.isInMouseFunctionMode = true;
	glfwSetInputMode(Window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN); // Set cursor to hidden.
	//orthogonalView();
	drawPicture();
	
	bool returnFlag = setMouseMuscleAttributes();
	
}

/*
 Puts the user in line muscle adjustment mode.
*/
void mouseAdjustMusclesLineMode()
{
	mouseFunctionsOff();
	Simulation.isPaused = true;
	Simulation.isInAdjustMuscleLineMode = true;
	Simulation.isInMouseFunctionMode = true;
	glfwSetInputMode(Window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN); // Set cursor to hidden.
	//orthogonalView();
	drawPicture();
	
	bool returnFlag = setMouseMuscleAttributes();
	
}

/*
 Puts the user in identify node mode.
*/
void mouseIdentifyNodeMode()
{
	mouseFunctionsOff();
	Simulation.isPaused = true;
	Simulation.isInFindNodeMode = true;
	Simulation.isInMouseFunctionMode = true;
	glfwSetInputMode(Window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN); // Set cursor to hidden.
	//orthogonalView();
	drawPicture();
}

/*
	Calls the functions that get user inputs for modifying the refractory periods 
	and conduction velocities of the selected muscles
*/
bool setMouseMuscleAttributes()
{
	// These functions now just set default values
	RefractoryPeriodAdjustmentMultiplier = 1.0;
	MuscleConductionVelocityAdjustmentMultiplier = 1.0;
	return(true);
}

/*
 This function sets up a node (nodeId) to be an ectopic beat node.
*/
void setEctopicBeat(int nodeId)
{
	Node[nodeId].isBeatNode = true;
	
	if(!Node[nodeId].isAblated)
	{
		Node[nodeId].isDrawNode = true;
		Node[nodeId].color.x = 1.0;
		Node[nodeId].color.y = 1.0;
		Node[nodeId].color.z = 0.0;
	}
	drawPicture();
	
	// Set default values - these used to come from user input functions
	Node[nodeId].beatPeriod = BeatPeriod; // Default to same as main beat
	Node[nodeId].beatTimer = 0; // Default to start immediately
	
	
	// We only let you set 1 ectopic beat at a time.
	Simulation.isInEctopicBeatMode = false;
}

/*
 This function is used to clear the print buffer.
*/
void clearStdin()
{
    int c;
    while ((c = getchar()) != '\n' && c != EOF)
    {
        /* discard characters */
    }
}


/*
 This function returns a timestamp in M-D-Y-H.M.S format.
 This is use so each file that is created has a unique name. 
 Note: You cannot create more than one file in a second or you will over write the previous file.
*/
string getTimeStamp()
{
	// Want to get a time stamp string representing current date/time, so we have a
	// unique name for each video/screenshot taken.
	time_t t = time(0); 
	struct tm * now = localtime( & t );
	int month = now->tm_mon + 1, day = now->tm_mday, year = now->tm_year, 
				curTimeHour = now->tm_hour, curTimeMin = now->tm_min, curTimeSec = now->tm_sec;

	stringstream smonth, sday, syear, stimeHour, stimeMin, stimeSec;

	smonth << month;
	sday << day;
	syear << (year + 1900); // The computer starts counting from the year 1900, so 1900 is year 0. So we fix that.
	stimeHour << curTimeHour;
	stimeMin << curTimeMin;
	stimeSec << curTimeSec;
	string timeStamp;

	if (curTimeMin <= 9)
	{
		timeStamp = smonth.str() + "-" + sday.str() + "-" + syear.str() + '_' + stimeHour.str() + ".0" + stimeMin.str() + 
					"." + stimeSec.str();
	}
	else
	{		
		timeStamp = smonth.str() + "-" + sday.str() + '-' + syear.str() + "_" + stimeHour.str() + "." + stimeMin.str() +
					"." + stimeSec.str();
	}

	return timeStamp;
}

/*
 This function turns the movie capture on.
*/
void movieOn()
{
	string ts = getTimeStamp();
	ts.append(".mp4");

	// Setting up the movie buffer.
	/*const char* cmd = "ffmpeg -loglevel quiet -r 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
		      "-threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip output.mp4";*/

	char baseCommand[512]; // Command to run ffmpeg with the correct parameters for capturing a movie
	//use sprintf to create the command string for ffmpeg, used XWindowSize and YWindowSize to set the size of the image

	//Low Quality, Fast Speed, Small Size
	// sprintf(baseCommand, "ffmpeg -loglevel quiet -r 60 -f rawvideo -pix_fmt rgba -s %dx%d -i - "
	// 	"-c:v libx264 -threads 0 -preset fast -y -pix_fmt yuv420p -crf 0 -vf vflip \"%s\"", XWindowSize, YWindowSize, ts.c_str());

	//Medium Quality, Medium Speed, Medium Size
	// sprintf(baseCommand, "ffmpeg -loglevel quiet -r 60 -f rawvideo -pix_fmt rgba -s %dx%d -i - "
	// 			"-c:v libx264 -threads 0 -preset medium -y -pix_fmt yuv420p -crf 0 -vf vflip \"%s\"", XWindowSize, YWindowSize, ts.c_str());

	//Max Quality, Low Speed, Large Size (change crf to 0 range[0,51] for lossless compression, but I wanted to keep the file size down)
	sprintf(baseCommand, "ffmpeg -loglevel quiet -r 60 -f rawvideo -pix_fmt rgba -s %dx%d -i - "
		"-c:v libx264 -threads 0 -preset veryslow -y -crf 16 -tune film -vf vflip \"%s\"", XWindowSize, YWindowSize, ts.c_str());

	//use the command string to create the output file name
	MovieFile = popen(baseCommand, "w");

	//Buffer = new int[XWindowSize*YWindowSize];
	Buffer = (unsigned char*)malloc(4* XWindowSize* YWindowSize);

	Simulation.isRecording = true;
}

/*
 This function turns the movie capture off.
*/
void movieOff()
{
	if(Simulation.isRecording) 
	{
		pclose(MovieFile);
	}
	free(Buffer);
	Simulation.isRecording = false;
}

/*
 This function takes a screen shot of the simulation.
*/
void screenShot()
{	
	bool savedPauseState;
	FILE* ScreenShotFile;
	unsigned char* buffer; //unsigned char because we are using RGBA data, which is 4 bytes per pixel, 1 char = 1 byte

    char cmd[512]; // Command to run ffmpeg with the correct parameters for capturing a screenshot

	//commands for ffmpeg, used XWindowSize and YWindowSize to set the size of the image
    sprintf(cmd, "ffmpeg -loglevel quiet -framerate 60 -f rawvideo -pix_fmt rgba -s %dx%d -i - "
                "-c:v libx264rgb -threads 0 -preset fast -y -crf 0 -vf vflip output1.mp4",
                XWindowSize, YWindowSize);

	
	//const char* cmd = "ffmpeg -r 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
	//              "-threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip output1.mp4";
	
	//open the pipe to ffmpeg and allocate the buffer for the screenshot with the size of 4*XWin* YWin to hold the RGBA data
	ScreenShotFile = popen(cmd, "w");
	buffer = (unsigned char*)malloc(4 * XWindowSize*YWindowSize*sizeof(int));
	
	if(!Simulation.isPaused) //if the simulation is running
	{
		Simulation.isPaused = true; //pause the simulation
		savedPauseState = false; //save the pause state
	}
	else //if the simulation is already paused
	{
		savedPauseState = true; //save the pause state
	}
	
	for(int i =0; i < 1; i++)
	{
		drawPicture();
		glReadPixels(0, 0, XWindowSize, YWindowSize, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
		fwrite(buffer, 4*XWindowSize*YWindowSize, 1, ScreenShotFile);
	}
	
	pclose(ScreenShotFile);
	free(buffer);

	string ts = getTimeStamp(); // Only storing in a separate variable for debugging purposes.
	string s = "ffmpeg -loglevel quiet -i output1.mp4 -qscale:v 1 -qmin 1 -qmax 1 " + ts + ".jpeg";
	// Convert back to a C-style string.
	const char *ccx = s.c_str();
	system(ccx);
	system("rm output1.mp4");
	printf("\nScreenshot Captured: \n");
	cout << "Saved as " << ts << ".jpeg" << endl;

	
	//system("ffmpeg -i output1.mp4 screenShot.jpeg");
	//system("rm output1.mp4");

	Simulation.isPaused = savedPauseState; //restore the pause state before we took the screenshot
	//ffmpeg -i output1.mp4 output_%03d.jpeg
}


/*
 This function saves all the node and muscle values set in the run to a file. This file can then be used at a
 later date to start a run with the exact settings used at the time of capture.
 So if the user has spent a great deal of time setting up a scenario, they can save the scenario and use it again later.
 We use it to create scenarios that have arrhythmias preprogrammed into them and have members from a class we are
 presenting to come up and see if they can use the ablation tool to eliminate the arythmia.
*/
void saveSettings()
{
	// Copying the latest node and muscle information down from the GPU.
	cudaMemcpy( Node, NodeGPU, NumberOfNodes*sizeof(nodeAttributesStructure), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy( Muscle, MuscleGPU, NumberOfMuscles*sizeof(muscleAttributesStructure), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Moving into the file that contains previuos run files.
	chdir("./PreviousRunsFile");
	   	
	// Creating an output file name to store run settings infomation in. It is unique down to the second to keep the user from 
	// overwriting files (You just cannot save more than one file a second).
	time_t t = time(0); 
	struct tm * now = localtime( & t );
	int month = now->tm_mon + 1, day = now->tm_mday, curTimeHour = now->tm_hour, curTimeMin = now->tm_min, curTimeSec = now->tm_sec;
	stringstream smonth, sday, stimeHour, stimeMin, stimeSec;

	smonth << month;
	sday << day;
	stimeHour << curTimeHour;
	stimeMin << curTimeMin;
	stimeSec << curTimeSec;
	string monthday;
	
	// We are using <=9 below to keep the minutes and seconds in a two-digit format. 
	// For example, 2 seconds would be displayed as 02 seconds.
	if(curTimeMin <= 9)
	{
		if(curTimeSec <= 9) monthday = smonth.str() + "-" + sday.str() + "-" + stimeHour.str() + ":0" + stimeMin.str() + ":0" + stimeSec.str();
		else monthday = smonth.str() + "-" + sday.str() + "-" + stimeHour.str() + ":0" + stimeMin.str() + ":" + stimeSec.str();
	}
	else monthday = smonth.str() + "-" + sday.str() + "-" + stimeHour.str() + ":" + stimeMin.str() + ":" + stimeSec.str();

	string timeStamp = "Run:" + monthday;
	const char *directoryName = timeStamp.c_str();
	
	// Creating the diretory to hold the run settings.
	if(mkdir(directoryName, 0777) == 0)
	{
		printf("\n Directory '%s' created successfully.\n", directoryName);
	}
	else
	{
		printf("\n Error creating directory '%s'.\n", directoryName);
	}
	
	// Moving into the directory
	chdir(directoryName);
	
	// Copying all the nodes and muscle (with their properties) into this folder in the file named run.
	FILE *settingFile;
  	settingFile = fopen("run", "wb");
  	fwrite(&PulsePointNode, sizeof(int), 1, settingFile);
  	fwrite(&UpNode, sizeof(int), 1, settingFile);
  	fwrite(&FrontNode, sizeof(int), 1, settingFile);
  	fwrite(&NumberOfNodes, sizeof(int), 1, settingFile);
  	fwrite(&NumberOfMuscles, sizeof(int), 1, settingFile);
  	fwrite(&RadiusOfLeftAtrium, sizeof(double), 1, settingFile);
  	int linksPerNode = MUSCLES_PER_NODE;
  	fwrite(&linksPerNode, sizeof(int), 1, settingFile);
  	fwrite(Node, sizeof(nodeAttributesStructure), NumberOfNodes, settingFile);
  	fwrite(Muscle, sizeof(muscleAttributesStructure), NumberOfMuscles, settingFile);
	fclose(settingFile);
	
	//Copying the simulationSetup file into this directory so you will know how it was initally setup.
	FILE *fileIn;
	FILE *fileOut;
	long sizeOfFile;
  	char *buffer;

	fileIn = fopen("../../simulationSetup", "rb");

	if(fileIn == NULL)
	{
		printf("\n\n The simulationSetup file does not exist\n\n");
		exit(0);
	}

	// Finding the size of the simulationSetup file.
	fseek (fileIn , 0 , SEEK_END);
  	sizeOfFile = ftell(fileIn);
  	rewind (fileIn);
  	
  	// Creating a buffer to hold the simulationSetup file.
  	buffer = (char*)malloc(sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, fileIn);
	fileOut = fopen("simulationSetup", "wb");
	fwrite (buffer, 1, sizeOfFile, fileOut);
	fclose(fileIn);
	fclose(fileOut);
	free(buffer);
	
	// Making a readMe file to put any infomation about why you are saving this run.
	system("gedit readMe");
	
	// Moving back to the SVT directory.
	chdir("../");
}

/*
 This function directs the action that needs to be taken if a user hits a key on the key board.
 The terminal screen lists out all the keys and what they will do.
*/
void KeyPressed(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    // Only process key press events, not releases or repeats

	// if (action != GLFW_PRESS)
	// 	return;

	// Get ImGui IO to check if it's capturing input
	ImGuiIO& io = ImGui::GetIO();
    
    // If ImGui is handling this event, return
    if (io.WantCaptureKeyboard)
        return;


    float dAngle = 0.01;
    float zoom = 0.01*RadiusOfLeftAtrium;
    float temp;
    float4 lookVector;
    float d;
    float4 centerOfMass;
    
    //copyNodesMusclesFromGPU();
    
    lookVector.x = CenterX - EyeX;
    lookVector.y = CenterY - EyeY;
    lookVector.z = CenterZ - EyeZ;
    d = sqrt(lookVector.x*lookVector.x + lookVector.y*lookVector.y + lookVector.z*lookVector.z);
    
    if(d < 0.00001)
    {
        printf("\n lookVector is too small\n");
        printf("\n Good Bye\n");
        exit(0);
    }
    else
    {
        lookVector.x /= d;
        lookVector.y /= d;
        lookVector.z /= d;
    }
    
    
    // WASD movement keys
    if(key == GLFW_KEY_W)  // Rotate counterclockwise on the x-axis
    {
		copyNodesFromGPU();
        centerOfMass = findCenterOfMass();
        for(int i = 0; i < NumberOfNodes; i++)
        {
            Node[i].position.x -= centerOfMass.x;
            Node[i].position.y -= centerOfMass.y;
            Node[i].position.z -= centerOfMass.z;
            temp = cos(dAngle)*Node[i].position.y - sin(dAngle)*Node[i].position.z;
            Node[i].position.z  = sin(dAngle)*Node[i].position.y + cos(dAngle)*Node[i].position.z;
            Node[i].position.y  = temp;
            Node[i].position.x += centerOfMass.x;
            Node[i].position.y += centerOfMass.y;
            Node[i].position.z += centerOfMass.z;
        }
        drawPicture();
        AngleOfSimulation.x += dAngle;
		copyNodesToGPU();
    }
    
    if(key == GLFW_KEY_S)  // Rotate clockwise on the x-axis
    {
		copyNodesFromGPU();
        centerOfMass = findCenterOfMass();
        for(int i = 0; i < NumberOfNodes; i++)
        {
            Node[i].position.x -= centerOfMass.x;
            Node[i].position.y -= centerOfMass.y;
            Node[i].position.z -= centerOfMass.z;
            temp = cos(-dAngle)*Node[i].position.y - sin(-dAngle)*Node[i].position.z;
            Node[i].position.z  = sin(-dAngle)*Node[i].position.y + cos(-dAngle)*Node[i].position.z;
            Node[i].position.y  = temp; 
            Node[i].position.x += centerOfMass.x;
            Node[i].position.y += centerOfMass.y;
            Node[i].position.z += centerOfMass.z;
        }
        drawPicture();
        AngleOfSimulation.x -= dAngle;
		copyNodesToGPU();
    }
    
    if(key == GLFW_KEY_D)  // Rotate counterclockwise on the y-axis
    {
		copyNodesFromGPU();
        centerOfMass = findCenterOfMass();
        for(int i = 0; i < NumberOfNodes; i++)
        {
            Node[i].position.x -= centerOfMass.x;
            Node[i].position.y -= centerOfMass.y;
            Node[i].position.z -= centerOfMass.z;
            temp =  cos(-dAngle)*Node[i].position.x + sin(-dAngle)*Node[i].position.z;
            Node[i].position.z  = -sin(-dAngle)*Node[i].position.x + cos(-dAngle)*Node[i].position.z;
            Node[i].position.x  = temp;
            Node[i].position.x += centerOfMass.x;
            Node[i].position.y += centerOfMass.y;
            Node[i].position.z += centerOfMass.z;
        }
        drawPicture();
        AngleOfSimulation.y -= dAngle;
		copyNodesToGPU();
    }
    
    if(key == GLFW_KEY_A)  // Rotate clockwise on the y-axis
    {
		copyNodesFromGPU();
        centerOfMass = findCenterOfMass();
        for(int i = 0; i < NumberOfNodes; i++)
        {
            Node[i].position.x -= centerOfMass.x;
            Node[i].position.y -= centerOfMass.y;
            Node[i].position.z -= centerOfMass.z;
            temp = cos(dAngle)*Node[i].position.x + sin(dAngle)*Node[i].position.z;
            Node[i].position.z  = -sin(dAngle)*Node[i].position.x + cos(dAngle)*Node[i].position.z;
            Node[i].position.x  = temp;
            Node[i].position.x += centerOfMass.x;
            Node[i].position.y += centerOfMass.y;
            Node[i].position.z += centerOfMass.z;
        }
        drawPicture();
        AngleOfSimulation.y += dAngle;
		copyNodesToGPU();
    }
    
    if(key == GLFW_KEY_Z)
    {
		copyNodesFromGPU();
        if(mods & GLFW_MOD_SHIFT)  // Uppercase Z - Rotate clockwise on the z-axis
        {
            centerOfMass = findCenterOfMass();
            for(int i = 0; i < NumberOfNodes; i++)
            {
                Node[i].position.x -= centerOfMass.x;
                Node[i].position.y -= centerOfMass.y;
                Node[i].position.z -= centerOfMass.z;
                temp = cos(-dAngle)*Node[i].position.x - sin(-dAngle)*Node[i].position.y;
                Node[i].position.y  = sin(-dAngle)*Node[i].position.x + cos(-dAngle)*Node[i].position.y;
                Node[i].position.x  = temp;
                Node[i].position.x += centerOfMass.x;
                Node[i].position.y += centerOfMass.y;
                Node[i].position.z += centerOfMass.z;
            }
            drawPicture();
            AngleOfSimulation.z -= dAngle;
        }
        else  // Lowercase z - Rotate counterclockwise on the z-axis
        {
            centerOfMass = findCenterOfMass();
            for(int i = 0; i < NumberOfNodes; i++)
            {
                Node[i].position.x -= centerOfMass.x;
                Node[i].position.y -= centerOfMass.y;
                Node[i].position.z -= centerOfMass.z;
                temp = cos(dAngle)*Node[i].position.x - sin(dAngle)*Node[i].position.y;
                Node[i].position.y  = sin(dAngle)*Node[i].position.x + cos(dAngle)*Node[i].position.y;
                Node[i].position.x  = temp;
                Node[i].position.x += centerOfMass.x;
                Node[i].position.y += centerOfMass.y;
                Node[i].position.z += centerOfMass.z;
            }
            drawPicture();
            AngleOfSimulation.z += dAngle;
        }
		copyNodesToGPU();
    }
    
    if(key == GLFW_KEY_E)
    {
		copyNodesFromGPU();
        if(mods & GLFW_MOD_SHIFT)  // Uppercase E - Zoom out
        {
            for(int i = 0; i < NumberOfNodes; i++)
            {
                Node[i].position.x += zoom*lookVector.x;
                Node[i].position.y += zoom*lookVector.y;
                Node[i].position.z += zoom*lookVector.z;
            }
            CenterOfSimulation.x += zoom*lookVector.x;
            CenterOfSimulation.y += zoom*lookVector.y;
            CenterOfSimulation.z += zoom*lookVector.z;
            drawPicture();
        }
        else  // Lowercase e - Zoom in
        {
            for(int i = 0; i < NumberOfNodes; i++)
            {
                Node[i].position.x -= zoom*lookVector.x;
                Node[i].position.y -= zoom*lookVector.y;
                Node[i].position.z -= zoom*lookVector.z;
            }
            CenterOfSimulation.x -= zoom*lookVector.x;
            CenterOfSimulation.y -= zoom*lookVector.y;
            CenterOfSimulation.z -= zoom*lookVector.z;
            drawPicture();
        }
		copyNodesToGPU();
    }
    
}

/*
 This function is called when the mouse moves without any button pressed.
 x and y are the current mouse coordinates.
 x come in as (0, XWindowSize) and y comes in as (0, YWindowSize). 
 We translates them to MouseX (-1, 1) and MouseY (-1, 1) to corospond to the openGL window size.
 We then use MouseX and MouseY to determine where the mouse is in the simulation.
*/
void mousePassiveMotionCallback(GLFWwindow* window, double x, double y)
{
	MouseX = ( 2.0*x/XWindowSize - 1.0)*RadiusOfLeftAtrium;
	MouseY = (-2.0*y/YWindowSize + 1.0)*RadiusOfLeftAtrium;
}

/*
 This function does an action based on the mode the viewer is in and which mouse button the user pressed.
*/
void myMouse(GLFWwindow* window, int button, int action, int mods)
{	

	//Add this if we want the GUI to only accept GUI handling until you ckick off of it
    // Get ImGui IO to check if it's capturing input
    ImGuiIO& io = ImGui::GetIO();
    
    // If ImGui is handling this mouse event, return
    if (io.WantCaptureMouse)
        return;
	
	float d, dx, dy, dz;
	float hit;
	int muscleId;
	
	if(action == GLFW_PRESS)
	{
		copyNodesMusclesFromGPU();
		hit = HitMultiplier*RadiusOfLeftAtrium;
		
		if(button == GLFW_MOUSE_BUTTON_LEFT)
		{	
			if(Simulation.isInAdjustMuscleLineMode)
			{
				// Finding the two closest nodes to the mouse.
				int nodeId1 = -1;
				int nodeId2 = -1;
				int connectingMuscle = -1;
				int test = -1;
				float minDistance = 2.0*RadiusOfLeftAtrium;
				for(int i = 0; i < NumberOfNodes; i++)
				{
					dx = MouseX - Node[i].position.x;
					dy = MouseY - Node[i].position.y;
					dz = MouseZ - Node[i].position.z;
					d = sqrt(dx*dx + dy*dy + dz*dz);
					if(d < minDistance)
					{
						minDistance = d;
						nodeId2 = nodeId1;
						nodeId1 = i;
					}
				}
				
				// If for some reason two nodes were not found. Not sure how this could
				// happen, but just to be safe we put a check in here.
				if(nodeId2 == -1)
				{
					printf("\n Two nodes were not found try again.\n");
					printf("\n MouseZ = %lf.\n", MouseZ);
				}
				// We got the two closest nodes to the mouse. Now see if there is a muscle that
				// connects these two nodes. If there is a connecting muscle, adjust it.
				else
				{
					if(!Node[nodeId1].isAblated)
					{
						Node[nodeId1].color.x = 1.0;
						Node[nodeId1].color.y = 0.0;
						Node[nodeId1].color.z = 1.0;
						Node[nodeId1].isDrawNode = true;
					}
					
					if(!Node[nodeId2].isAblated)
					{
						Node[nodeId2].color.x = 1.0;
						Node[nodeId2].color.y = 0.0;
						Node[nodeId2].color.z = 1.0;
						Node[nodeId2].isDrawNode = true;
					}
					
					for(int i = 0; i < MUSCLES_PER_NODE; i++) // Spinnning through muscles on node 1.
					{
						muscleId = Node[nodeId1].muscle[i]; 
						if(muscleId != -1)
						{
							for(int j = 0; j < MUSCLES_PER_NODE; j++) // Spinnning through muscles on node 2.
							{
								test = Node[nodeId2].muscle[j];
								if(muscleId == test) // Checking to see if we get a match.
								{
									connectingMuscle = muscleId;
								}
							}
						}
					}
					if(connectingMuscle == -1)
					{
						printf("\n No connecting muscle was found try again.\n");
					}
					else
					{
						muscleId = connectingMuscle;
						Muscle[muscleId].refractoryPeriod = BaseMuscleRefractoryPeriod*RefractoryPeriodAdjustmentMultiplier;
						Muscle[muscleId].conductionVelocity = BaseMuscleConductionVelocity*MuscleConductionVelocityAdjustmentMultiplier;
						Muscle[muscleId].conductionDuration = Muscle[muscleId].naturalLength/Muscle[muscleId].conductionVelocity;
						Muscle[muscleId].color.x = 1.0;
						Muscle[muscleId].color.y = 0.0;
						Muscle[muscleId].color.z = 1.0;
						Muscle[muscleId].color.w = 0.0;
						
						checkMuscle(muscleId);		
					}
				}
			}
			else
			{
				for(int i = 0; i < NumberOfNodes; i++)
				{
					dx = MouseX - Node[i].position.x;
					dy = MouseY - Node[i].position.y;
					dz = MouseZ - Node[i].position.z;
					
					if(sqrt(dx*dx + dy*dy + dz*dz) < hit)
					{
						if(Simulation.isInAblateMode)
						{
							Node[i].isAblated = true;
							Node[i].isDrawNode = true;
							Node[i].color.x = 1.0;
							Node[i].color.y = 1.0;
							Node[i].color.z = 1.0;
						}
						
						if(Simulation.isInEctopicBeatMode)
						{
							Simulation.isPaused = true;
							// printf("\n Node number = %d", i);
							setEctopicBeat(i);
						}
						
						if(Simulation.isInAdjustMuscleAreaMode)
						{
							for(int j = 0; j < MUSCLES_PER_NODE; j++)
							{
								muscleId = Node[i].muscle[j];
								if(muscleId != -1)
								{
									// This sets the muscle to the base value then adjusts it. 
									Muscle[muscleId].refractoryPeriod = BaseMuscleRefractoryPeriod*RefractoryPeriodAdjustmentMultiplier;
									Muscle[muscleId].conductionVelocity = BaseMuscleConductionVelocity*MuscleConductionVelocityAdjustmentMultiplier;
									
									// This adjusts the muscle based on its current value.
									//Muscle[muscleId].refractoryPeriod *= RefractoryPeriodAdjustmentMultiplier;
									//Muscle[muscleId].conductionVelocity *= MuscleConductionVelocityAdjustmentMultiplier;
									
									Muscle[muscleId].conductionDuration = Muscle[muscleId].naturalLength/Muscle[muscleId].conductionVelocity;
									Muscle[muscleId].color.x = 1.0;
									Muscle[muscleId].color.y = 0.0;
									Muscle[muscleId].color.z = 1.0;
									Muscle[muscleId].color.w = 0.0;
									
									checkMuscle(muscleId);
								}
							}
							
							Node[i].isDrawNode = true;
							if(!Node[i].isAblated) // If it is not ablated color it.
							{
								Node[i].color.x = 0.0;
								Node[i].color.y = 0.0;
								Node[i].color.z = 1.0;
							}
						}
						
						if(Simulation.isInEctopicEventMode)
						{
							cudaMemcpy( Node, NodeGPU, NumberOfNodes*sizeof(nodeAttributesStructure), cudaMemcpyDeviceToHost);
							cudaErrorCheck(__FILE__, __LINE__);
							
							Node[i].isFiring = true; // Setting the ith node to fire the next time in the next time step.
							
							cudaMemcpy( NodeGPU, Node, NumberOfNodes*sizeof(nodeAttributesStructure), cudaMemcpyHostToDevice );
							cudaErrorCheck(__FILE__, __LINE__);
						}
						
						if(Simulation.isInFindNodeMode)
						{
							Node[i].isDrawNode = true;
							Node[i].color.x = 1.0;
							Node[i].color.y = 0.0;
							Node[i].color.z = 1.0;
							// printf("\n Node number = %d", i);
						}
					}
				}
			}
		}
		else if(button == GLFW_MOUSE_BUTTON_RIGHT) // Right Mouse button down
		{
			if(Simulation.isInAdjustMuscleLineMode)
			{
				// Finding the two closest nodes to the mouse.
				int nodeId1 = -1;
				int nodeId2 = -1;
				int connectingMuscle = -1;
				int test = -1;
				float minDistance = 2.0*RadiusOfLeftAtrium;
				for(int i = 0; i < NumberOfNodes; i++)
				{
					dx = MouseX - Node[i].position.x;
					dy = MouseY - Node[i].position.y;
					dz = MouseZ - Node[i].position.z;
					d = sqrt(dx*dx + dy*dy + dz*dz);
					if(d < minDistance)
					{
						minDistance = d;
						nodeId2 = nodeId1;
						nodeId1 = i;
					}
				}
				
				// If for some reason two nodes were not found. Not sure how this could
				// happen, but just to be safe we put a check in here.
				if(nodeId2 == -1)
				{
					printf("\n Two nodes were not found try again.\n");
					printf("\n MouseZ = %lf.\n", MouseZ);
				}
				// We got the two closest nodes to the mouse. Now see if there is a muscle that
				// connects these two nodes. If there is a connecting muscle, adjust it.
				else
				{
					if(!Node[nodeId1].isAblated)
					{
						Node[nodeId1].color.x = 0.0;
						Node[nodeId1].color.y = 1.0;
						Node[nodeId1].color.z = 0.0;
						Node[nodeId1].isDrawNode = false;
					}
					
					if(!Node[nodeId2].isAblated)
					{
						Node[nodeId2].color.x = 0.0;
						Node[nodeId2].color.y = 1.0;
						Node[nodeId2].color.z = 0.0;
						Node[nodeId2].isDrawNode = false;
					}
					
					for(int i = 0; i < MUSCLES_PER_NODE; i++) // Spinnning through muscles on node 1.
					{
						muscleId = Node[nodeId1].muscle[i]; 
						if(muscleId != -1)
						{
							for(int j = 0; j < MUSCLES_PER_NODE; j++) // Spinnning through muscles on node 2.
							{
								test = Node[nodeId2].muscle[j];
								if(muscleId == test) // Checking to see if we get a match.
								{
									connectingMuscle = muscleId;
								}
							}
						}
					}
					if(connectingMuscle == -1)
					{
						printf("\n No connecting muscle was found try again.\n");
					}
					else
					{
						muscleId = connectingMuscle;
						Muscle[muscleId].refractoryPeriod = BaseMuscleRefractoryPeriod;
						Muscle[muscleId].conductionVelocity = BaseMuscleConductionVelocity;
						Muscle[muscleId].conductionDuration = Muscle[muscleId].naturalLength/Muscle[muscleId].conductionVelocity;
						Muscle[muscleId].color.x = 0.0;
						Muscle[muscleId].color.y = 1.0;
						Muscle[muscleId].color.z = 0.0;
						Muscle[muscleId].color.w = 0.0;
						// Turning the muscle back on if it was disabled.
						Muscle[muscleId].isEnabled = true;
						
						checkMuscle(muscleId);		
					}
				}
			}
			else
			{
				for(int i = 0; i < NumberOfNodes; i++)
				{
					dx = MouseX - Node[i].position.x;
					dy = MouseY - Node[i].position.y;
					dz = MouseZ - Node[i].position.z;
					if(sqrt(dx*dx + dy*dy + dz*dz) < hit)
					{
						if(Simulation.isInAblateMode)
						{
							Node[i].isAblated = false;
							Node[i].isDrawNode = false;
							Node[i].color.x = 0.0;
							Node[i].color.y = 1.0;
							Node[i].color.z = 0.0;
						}
						
						if(Simulation.isInAdjustMuscleAreaMode)
						{
							for(int j = 0; j < MUSCLES_PER_NODE; j++)
							{
								muscleId = Node[i].muscle[j];
								if(muscleId != -1)
								{
									Muscle[muscleId].refractoryPeriod = BaseMuscleRefractoryPeriod;
									Muscle[muscleId].conductionVelocity = BaseMuscleConductionVelocity;
									Muscle[muscleId].conductionDuration = Muscle[muscleId].naturalLength/Muscle[muscleId].conductionVelocity;
									Muscle[muscleId].color.x = 0.0;
									Muscle[muscleId].color.y = 1.0;
									Muscle[muscleId].color.z = 0.0;
									Muscle[muscleId].color.w = 0.0;
									
									// Turning the muscle back on if it was disabled.
									Muscle[muscleId].isEnabled = true;
									
									// Checking to see if the muscle needs to be killed.
									checkMuscle(muscleId);
								}
							}
							
							Node[i].isDrawNode = true;
							if(!Node[i].isAblated) // If it is not ablated color it.
							{
								Node[i].color.x = 0.0;
								Node[i].color.y = 1.0;
								Node[i].color.z = 0.0;
							}
						}
					}
				}
			}
		}
		else if(button == GLFW_MOUSE_BUTTON_MIDDLE)
		{
			if(ScrollSpeedToggle == 0)
			{
				ScrollSpeedToggle = 1;
				ScrollSpeed = 1.0;
				// printf("\n speed = %f\n", ScrollSpeed);
			}
			else
			{
				ScrollSpeedToggle = 0;
				ScrollSpeed = 0.1;
				// printf("\n speed = %f\n", ScrollSpeed);
			}
			
		}
		drawPicture();
		copyNodesMusclesToGPU();
		//printf("\nSNx = %f SNy = %f SNz = %f\n", NodePosition[0].x, NodePosition[0].y, NodePosition[0].z);
	}
	
}

void scrollWheel(GLFWwindow* window, double xoffset, double yoffset)
{
    if(yoffset > 0) // Scroll up
    {
        MouseZ -= ScrollSpeed;
    }
    else if(yoffset < 0) // Scroll down
    {
        MouseZ += ScrollSpeed;
    }
    // printf("MouseZ = %f\n", MouseZ);
    drawPicture();
}
