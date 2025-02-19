/*
 This file contains all the callBack functions and function that it calls to do its work.
 In isense this file contains all the ways a user can interact (Mouse and Ternimal) with 
 a running simulation.
 
 The functions in this file are listed below and in this order.
 
 void Display(void);
 void idle();
 void reshape(int, int);
 void mouseFunctionsOff();
 void mouseAblateMode();
 void mouseEctopicBeatMode();
 void mouseAdjustMusclesAreaMode();
 void mouseAdjustMusclesLineMode();
 void mouseIdentifyNodeMode();
 int setMouseMuscleAttributes();
 void setMouseMuscleRefractoryPeriod();
 void setMouseMuscleConductionVelocity();
 void setEctopicBeat(int nodeId);
 void clearStdin();
 void getEctopicBeatPeriod(int);
 void getEctopicBeatOffset(int);
 string getTimeStamp();
 void movieOn();
 void movieOff();
 void screenShot();
 void saveSettings();
 void KeyPressed(unsigned char, int, int);
 void mousePassiveMotionCallback(int, int);
 void mymouse(int, int, int, int);
*/

/*
 OpenGL callback when the window is created or reshaped.
*/
void Display(void)
{
	drawPicture();
}

/*
 OpenGL callback when the window is doing nothing else.
*/
void idle()
{
	nBody(Dt);
}

/*
 OpenGL callback when the window is reshaped.
*/
void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);
}

/*
 Turns off all the user interactions.
*/
void mouseFunctionsOff()
{
	PauseIs = true;
	AblateModeIs = false;
	EctopicBeatModeIs = false;
	EctopicEventModeIs = false;
	AdjustMuscleAreaModeIs = false;
	AdjustMuscleLineModeIs = false;
	FindNodeModeIs = false;
	MouseFunctionModeIs = false;
	terminalPrint();
	glutSetCursor(GLUT_CURSOR_DESTROY);
	drawPicture();
}

/*
 Puts the user in ablate mode.
*/
void mouseAblateMode()
{
	PauseIs = true;
	AblateModeIs = true;
	EctopicBeatModeIs = false;
	EctopicEventModeIs = false;
	AdjustMuscleAreaModeIs = false;
	AdjustMuscleLineModeIs = false;
	FindNodeModeIs = false;
	MouseFunctionModeIs = true;
	glutSetCursor(GLUT_CURSOR_NONE);
	//orthoganialView();
	terminalPrint();
	drawPicture();
}

/*
 Puts the user in extopic beat mode.
*/
void mouseEctopicBeatMode()
{
	PauseIs = true;
	AblateModeIs = false;
	EctopicBeatModeIs = true;
	EctopicEventModeIs = false;
	AdjustMuscleAreaModeIs = false;
	AdjustMuscleLineModeIs = false;
	FindNodeModeIs = false;
	MouseFunctionModeIs = true;
	glutSetCursor(GLUT_CURSOR_NONE);
	//orthoganialView();
	terminalPrint();
	drawPicture();
	system("clear");
	printf("\n You are in create ectopic beat mode.");
	printf("\n\n Use the mouse to select a node.");
	printf("\n");
}

/*
 Puts the user in extopic event mode.
*/
void mouseEctopicEventMode()
{
	PauseIs = true;
	AblateModeIs = false;
	EctopicBeatModeIs = false;
	EctopicEventModeIs = true;
	AdjustMuscleAreaModeIs = false;
	AdjustMuscleLineModeIs = false;
	FindNodeModeIs = false;
	MouseFunctionModeIs = true;
	glutSetCursor(GLUT_CURSOR_NONE);
	//orthoganialView();
	drawPicture();
	terminalPrint();
}

/*
 Puts the user in area muscle adjustment mode.
*/
void mouseAdjustMusclesAreaMode()
{
	PauseIs = true;
	AblateModeIs = false;
	EctopicBeatModeIs = false;
	EctopicEventModeIs = false;
	AdjustMuscleAreaModeIs = true;
	AdjustMuscleLineModeIs = false;
	FindNodeModeIs = false;
	MouseFunctionModeIs = true;
	glutSetCursor(GLUT_CURSOR_NONE);
	//orthoganialView();
	drawPicture();
	
	int returnCode = setMouseMuscleAttributes();
	
	if(returnCode == 1)
	{
		terminalPrint();
	}
}

/*
 Puts the user in line muscle adjustment mode.
*/
void mouseAdjustMusclesLineMode()
{
	PauseIs = true;
	AblateModeIs = false;
	EctopicBeatModeIs = false;
	EctopicEventModeIs = false;
	AdjustMuscleAreaModeIs = false;
	AdjustMuscleLineModeIs = true;
	FindNodeModeIs = false;
	MouseFunctionModeIs = true;
	glutSetCursor(GLUT_CURSOR_NONE);
	//orthoganialView();
	drawPicture();
	
	int returnCode = setMouseMuscleAttributes();
	
	if(returnCode == 1)
	{
		terminalPrint();
	}
}

/*
 Puts the user in identify node mode.
*/
void mouseIdentifyNodeMode()
{
	PauseIs = true;
	AblateModeIs = false;
	EctopicBeatModeIs = false;
	EctopicEventModeIs = false;
	AdjustMuscleAreaModeIs = false;
	AdjustMuscleLineModeIs = false;
	FindNodeModeIs = true;
	MouseFunctionModeIs = true;
	glutSetCursor(GLUT_CURSOR_NONE);
	//orthoganialView();
	drawPicture();
	terminalPrint();
}

/*
 Calls the functions that get user inputs for how to change selected muscles' refractory periods 
 and conduction velocities.
*/
int setMouseMuscleAttributes()
{
	setMouseMuscleRefractoryPeriod();
	setMouseMuscleConductionVelocity();
	return(1);
}

/*
 This function asks the user to type in the terminal screen the value to be multipled by the
 selected muscles' refractory period.
*/
void setMouseMuscleRefractoryPeriod()
{
	system("clear");
	RefractoryPeriodAdjustmentMultiplier = -1.0;
	
	printf("\n\n Enter the refractory period multiplier.");
	printf("\n A number greater than 1 will make it longer.");
	printf("\n A number between 0 and 1 will make it shorter.");
	printf("\n\n Refractory period multiplier = ");
	fflush(stdin);
	scanf("%f", &RefractoryPeriodAdjustmentMultiplier);
	if(RefractoryPeriodAdjustmentMultiplier < 0)
	{
		system("clear");
		printf("\n You cannot adjust the the refractory period by a negative number.");
		printf("\n Retry\n");
		setMouseMuscleRefractoryPeriod();
	}
}

/*
 This function asks the user to type in the terminal screen the value to be multipled by the
 selected muscles' conduction velocity.
*/
void setMouseMuscleConductionVelocity()
{
	system("clear");
	MuscleConductionVelocityAdjustmentMultiplier = -1.0;
	
	printf("\n\n Enter conduction velocity multipier.");
	printf("\n A number between 0 and 1 will slow it down.");
	printf("\n A number bigger than 1 will speed it up.");
	printf("\n\n Conduction velocity multiplier = ");
	fflush(stdin);
	scanf("%f", &MuscleConductionVelocityAdjustmentMultiplier);
	if(MuscleConductionVelocityAdjustmentMultiplier <= 0)
	{
		system("clear");
		printf("\n You cannot adjust the the conduction velocity by a nonpositive number.");
		printf("\n Retry\n");
		setMouseMuscleConductionVelocity();
	}
}

/*
 This function sets a node (nodeId) up to be an ectopic beat node.
*/
void setEctopicBeat(int nodeId)
{
	Node[nodeId].isBeatNode = true;
	
	if(Node[nodeId].isAblated == false)
	{
		Node[nodeId].drawNodeIs = true;
		Node[nodeId].color.x = 1.0;
		Node[nodeId].color.y = 1.0;
		Node[nodeId].color.z = 0.0;
	}
	drawPicture();
	
	getEctopicBeatPeriod(nodeId);
	getEctopicBeatOffset(nodeId);
	
	// We only let you set 1 ectopic beat at a time.
	EctopicBeatModeIs = false;
	terminalPrint();
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
 This function gets the ectopic beat period from the user.
*/
void getEctopicBeatPeriod(int nodeId)
{
	float period;
	fflush(stdin);
	system("clear");
	printf("\n The current driving beat Period = %f.", BeatPeriod);
	printf("\n Enter the period of your ectopic beat.");
	
	printf("\n\n Ectopic period = ");
	scanf("%f", &period);
	if(period <= 0)
	{
		system("clear");
		printf("\n You entered %f.", Node[nodeId].beatPeriod);
		printf("\n You cannot have a beat period that is a nonpositive number.");
		printf("\n Retry\n");
		getEctopicBeatPeriod(nodeId);
	}
	else
	{
		Node[nodeId].beatPeriod = period;
	}
	clearStdin();
}

/*
 This function gets the ectopic beat offset from the user. This is the amount of time the
 user wants to pause before turning on the ectopic beat. This allows the user to time the 
 ectopic beats relative to the current time. So the user can set beats to trigger at diffent 
 times.
*/
void getEctopicBeatOffset(int nodeId)
{
	system("clear");
	printf("\n The current Time into the beat is %f.", Node[nodeId].beatTimer);
	printf("\n Enter the time offset of your ectopic event.");
	printf("\n This will allow you to time your ectopic beat with the driving beat.");
	printf("\n Zero will start the ectopic beat now.");
	printf("\n A positive number will delay the ectopic beat by that amount.");
	printf("\n\n Ectopic time delay = ");
	fflush(stdin);
	float timeDelay;
	scanf("%f", &timeDelay);
	if(timeDelay < 0)
	{
		system("clear");
		printf("\n You cannot have a time delay that is a negative number.");
		printf("\n Retry\n");
		getEctopicBeatOffset(nodeId);
	}
	else
	{
		Node[nodeId].beatTimer = Node[nodeId].beatPeriod - timeDelay;
	}
}

/*
 This function returns a timestamp in M-D-Y-H.M.S format.
 This is use so each file that is created has a unuque name. 
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
		timeStamp = smonth.str() + "-" + sday.str() + "-" + syear.str() + '_' + stimeHour.str() + ".0" + stimeMin.str() + 
					"." + stimeSec.str();
	else			
		timeStamp = smonth.str() + "-" + sday.str() + '-' + syear.str() + "_" + stimeHour.str() + "." + stimeMin.str() +
					"." + stimeSec.str();
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

	string baseCommand = "ffmpeg -loglevel quiet -r 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
				"-c:v libx264rgb -threads 0 -preset fast -y -pix_fmt yuv420p -crf 0 -vf vflip ";

	string z = baseCommand + ts;

	const char *ccx = z.c_str();
	MovieFile = popen(ccx, "w");
	//Buffer = new int[XWindowSize*YWindowSize];
	Buffer = (int*)malloc(XWindowSize*YWindowSize*sizeof(int));
	MovieIsOn = true;
}

/*
 This function turns the movie capture off.
*/
void movieOff()
{
	if(MovieIsOn == true) 
	{
		pclose(MovieFile);
	}
	free(Buffer);
	MovieIsOn = false;
}

/*
 This function takes a screen shot of the simulation.
*/
void screenShot()
{	
	bool pauseFlagIs;
	FILE* ScreenShotFile;
	int* buffer;

	const char* cmd = "ffmpeg -loglevel quiet -framerate 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
				"-c:v libx264rgb -threads 0 -preset fast -y -crf 0 -vf vflip output1.mp4";
	//const char* cmd = "ffmpeg -r 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
	//              "-threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip output1.mp4";
	ScreenShotFile = popen(cmd, "w");
	buffer = (int*)malloc(XWindowSize*YWindowSize*sizeof(int));
	
	if(PauseIs == false) 
	{
		PauseIs = true;
		pauseFlagIs = false;
	}
	else
	{
		pauseFlagIs = true;
	}
	
	for(int i =0; i < 1; i++)
	{
		drawPicture();
		glReadPixels(5, 5, XWindowSize, YWindowSize, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
		fwrite(buffer, sizeof(int)*XWindowSize*YWindowSize, 1, ScreenShotFile);
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
	
	PauseIs = pauseFlagIs;
	//ffmpeg -i output1.mp4 output_%03d.jpeg
}

/*
 This function saves all the node and muscle values set in the run to a file. This file then be used at a
 latter date to start a run with exact settings used at the time of capture.
 So if the user has spent a great deal of time setting up a scenario they can be saved and used again latter.
 We use it to create senarios that have arythmias preprogramed into them and have members from a class we are
 presenting to come up and see if they can use the ablate tool to elemenate the arythmia.
*/
void saveSettings()
{
	cudaMemcpy( Node, NodeGPU, NumberOfNodes*sizeof(nodeAtributesStructure), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy( Muscle, MuscleGPU, NumberOfMuscles*sizeof(muscleAtributesStructure), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	chdir("./PreviousRunsFile");
	   	
	//Create output file name to store run settings in.
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
	if(curTimeMin <= 9)
		if(curTimeSec <= 9) monthday = smonth.str() + "-" + sday.str() + "-" + stimeHour.str() + ":0" + stimeMin.str() + ":0" + stimeSec.str();
		else monthday = smonth.str() + "-" + sday.str() + "-" + stimeHour.str() + ":0" + stimeMin.str() + ":" + stimeSec.str();
	else monthday = smonth.str() + "-" + sday.str() + "-" + stimeHour.str() + ":" + stimeMin.str() + ":" + stimeSec.str();
	string timeStamp = "Run:" + monthday;
	const char *diretoryName = timeStamp.c_str();
	
	if(mkdir(diretoryName, 0777) == 0)
	{
		printf("\n Directory '%s' created successfully.\n", diretoryName);
	}
	else
	{
		printf("\n Error creating directory '%s'.\n", diretoryName);
	}
	
	chdir(diretoryName);
	
	// Copying all the nodes and muscle (with their properties) into this folder in the file named run.
	FILE *settingFile;
  	settingFile = fopen("run", "wb");
  	fwrite(&PulsePointNode, sizeof(int), 1, settingFile);
  	fwrite(&UpNode, sizeof(int), 1, settingFile);
  	fwrite(&FrontNode, sizeof(int), 1, settingFile);
  	fwrite(&NumberOfNodes, sizeof(int), 1, settingFile);
  	fwrite(&NumberOfMuscles, sizeof(int), 1, settingFile);
  	int linksPerNode = MUSCLES_PER_NODE;
  	fwrite(&linksPerNode, sizeof(int), 1, settingFile);
  	fwrite(Node, sizeof(nodeAtributesStructure), NumberOfNodes, settingFile);
  	fwrite(Muscle, sizeof(muscleAtributesStructure), NumberOfMuscles, settingFile);
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
	fseek (fileIn , 0 , SEEK_END);
  	sizeOfFile = ftell(fileIn);
  	rewind (fileIn);
  	buffer = (char*)malloc(sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, fileIn);
	fileOut = fopen("simulationSetup", "wb");
	fwrite (buffer, 1, sizeOfFile, fileOut);
	fclose(fileIn);
	fclose(fileOut);
	
	// Making a readMe file to put any infomation about why you are saving this run.
	system("gedit readMe");
	
	// Moving back to the SVT directory.
	chdir("../");
}

/*
 This function directs the action that needs to be taken if a user hit a key on the key board.
 The terminal screen lists out all the keys and what they will do.
*/
void KeyPressed(unsigned char key, int x, int y)
{
	float dAngle = 0.01;
	float zoom = 0.01*RadiusOfLeftAtrium;
	float temp;
	float4 lookVector;
	float d;
	float4 centerOfMass;
	
	copyNodesMusclesFromGPU();
	
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
	
	if(key == 'h')  // Help menu
	{
		helpMenu();
	}
	
	if(key == 'q') // quit
	{
		glutDestroyWindow(Window);
		printf("\n Good Bye\n");
		exit(0);
	}
	
	if(key == 'r')  // Run toggle
	{
		if(PauseIs == false) PauseIs = true;
		else PauseIs = false;
		terminalPrint();
	}
	
	if(key == 'n')  // Draw nodes toggle
	{
		if(DrawNodesFlag == 0) DrawNodesFlag = 1;
		else if(DrawNodesFlag == 1) DrawNodesFlag = 2;
		else DrawNodesFlag = 0;
		drawPicture();
		terminalPrint();
	}
	
	if(key == 'g')  // Draw full or front half toggle
	{
		if(DrawFrontHalfFlag == 0) DrawFrontHalfFlag = 1;
		else DrawFrontHalfFlag = 0;
		drawPicture();
		terminalPrint();
	}
	
	if(key == 'B')  // Raising the beat period
	{
		Node[PulsePointNode].beatPeriod += 10;
		cudaMemcpy( NodeGPU, Node, NumberOfNodes*sizeof(nodeAtributesStructure), cudaMemcpyHostToDevice );
		cudaErrorCheck(__FILE__, __LINE__);
		terminalPrint();
	}
	if(key == 'b')  // Lowering the beat period
	{
		Node[PulsePointNode].beatPeriod -= 10;
		if(Node[PulsePointNode].beatPeriod < 0) 
		{
			Node[PulsePointNode].beatPeriod = 0;  // You don't want the beat to go negative
		}
		cudaMemcpy( NodeGPU, Node, NumberOfNodes*sizeof(nodeAtributesStructure), cudaMemcpyHostToDevice );
		cudaErrorCheck(__FILE__, __LINE__);
		terminalPrint();
	}
	
	if(key == 'v') // Orthoganal/Fulstrium view
	{
		if(ViewFlag == 0) 
		{
			ViewFlag = 1;
			fulstrumView();
		}
		else 
		{
			ViewFlag = 0;
			orthoganialView();
		}
		drawPicture();
		terminalPrint();
	}
	
	if(key == 'm')  // Movie on
	{
		if(MovieIsOn == false) 
		{
			MovieIsOn = true;
			movieOn();
		}
		else 
		{
			MovieIsOn = false;
			movieOff();
		}
		terminalPrint();
	}
	
	if(key == 'S')  // Screenshot
	{	
		screenShot();
		terminalPrint();
	}
	
	if(key == '0')
	{
		setView(0);
		drawPicture();
		terminalPrint();
	}
	if(key == '1')
	{
		setView(1);
		drawPicture();
		terminalPrint();
	}
	if(key == '2')
	{
		setView(2);
		drawPicture();
		terminalPrint();
	}
	if(key == '3')
	{
		setView(3);
		drawPicture();
		terminalPrint();
	}
	if(key == '4')
	{
		setView(4);
		drawPicture();
		terminalPrint();
	}
	if(key == '5')
	{
		setView(5);
		drawPicture();
		terminalPrint();
	}
	if(key == '6')
	{
		setView(6);
		drawPicture();
		terminalPrint();
	}
	if(key == '7')
	{
		setView(7);
		drawPicture();
		terminalPrint();
	}
	if(key == '8')
	{
		setView(8);
		drawPicture();
		terminalPrint();
	}
	if(key == '9')
	{
		setView(9);
		drawPicture();
		terminalPrint();
	}
	if(key == '?') // Finding front and top reference nodes.
	{
		float maxZ = -10000.0;
		float maxY = -10000.0;
		int indexZ = -1;
		int indexY = -1;
		
		for(int i = 0; i < NumberOfNodes; i++)
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
		
		Node[indexZ].color.x = 0.0;
		Node[indexZ].color.y = 0.0;
		Node[indexZ].color.z = 1.0;
		
		Node[indexY].color.x = 1.0;
		Node[indexY].color.y = 0.0;
		Node[indexY].color.z = 1.0;
		
		system("clear");
		printf("\n Front node index = %d\n", indexZ);
		printf("\n Top node index   = %d\n", indexY);
		
		drawPicture();
	}
	if(key == 'w')  // Rotate counterclockwise on the x-axis
	{
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
	}
	if(key == 's')  // Rotate clockwise on the x-axis
	{
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
	}
	if(key == 'd')  // Rotate counterclockwise on the y-axis
	{
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
	}
	if(key == 'a')  // Rotate clockwise on the y-axis
	{
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
	}
	if(key == 'z')  // Rotate counterclockwise on the z-axis
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
	if(key == 'Z')  // Rotate clockwise on the z-axis
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
	if(key == 'e')  // Zoom in
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
	if(key == 'E')  // Zoom out
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
	
	if(key == ')')  // All mouse functions are off (shift 0)
	{
		mouseFunctionsOff();
		MouseFunctionModeIs = false;
	}
	if(key == '!')  // Ablate is on (shift 1)
	{
		mouseAblateMode();
		MouseFunctionModeIs = true;
	}
	if(key == '@')  // Ectopic beat is on (shift 2)
	{
		mouseEctopicBeatMode();
		MouseFunctionModeIs = true;
	}
	if(key == '#')  // You are in ectopic single trigger mode. (shift 3)
	{
		mouseEctopicEventMode();
		MouseFunctionModeIs = true;
	}
	if(key == '$') // muscle adjustment is on (shift 4)
	{
		mouseAdjustMusclesAreaMode();
		MouseFunctionModeIs = true;
	}
	if(key == '%') // muscle adjustment is on (shift 4)
	{
		mouseAdjustMusclesLineMode();
		MouseFunctionModeIs = true;
	}
	if(key == '^')  // Find node is on (shift 5)
	{
		mouseIdentifyNodeMode();
		MouseFunctionModeIs = true;
	}
	
	if(key == ']')  
	{
		HitMultiplier += 0.005;
		terminalPrint();
		//printf("\n Your selection area = %f times the radius of atrium. \n", HitMultiplier);
	}
	if(key == '[')
	{
		HitMultiplier -= 0.005;
		if(HitMultiplier < 0.0) HitMultiplier = 0.0;
		terminalPrint();
		//printf("\n Your selection area = %f times the radius of atrium. \n", HitMultiplier);
	}
	
	if(key == 'c')  // Recenter the simulation
	{
		centerObject();
		drawPicture();
	}
	
	if(key == 'k')  // Save your current setting so you can start with this run in the future.
	{
		saveSettings();
	}
	
	copyNodesMusclesToGPU();
}

/*
 This function is called when the mouse moves without any button pressed.
 x and y are the current mouse coordinates.
 x come in as (0, XWindowSize) and y comes in as (0, YWindowSize). 
 We translates them to MouseX (-1, 1) and MouseY (-1, 1) to corospond to the openGL window size.
 We then use MouseX and MouseY to determine where the mouse is in the simulation.
*/
void mousePassiveMotionCallback(int x, int y) 
{
	MouseX = ( 2.0*x/XWindowSize - 1.0)*RadiusOfLeftAtrium;
	MouseY = (-2.0*y/YWindowSize + 1.0)*RadiusOfLeftAtrium;
}

/*
 This function does an action based on the mode the viewer is in and which mouse button the user pressed.
*/
void mymouse(int button, int state, int x, int y)
{	
	float d, dx, dy, dz;
	float hit;
	int muscleId;
	
	if(state == GLUT_DOWN)
	{
		copyNodesMusclesFromGPU();
		hit = HitMultiplier*RadiusOfLeftAtrium;
		
		if(button == GLUT_LEFT_BUTTON)
		{	
			if(AdjustMuscleLineModeIs == true)
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
					if(Node[nodeId1].isAblated == false)
					{
						Node[nodeId1].color.x = 1.0;
						Node[nodeId1].color.y = 0.0;
						Node[nodeId1].color.z = 1.0;
						Node[nodeId1].drawNodeIs = true;
					}
					
					if(Node[nodeId2].isAblated == false)
					{
						Node[nodeId2].color.x = 1.0;
						Node[nodeId2].color.y = 0.0;
						Node[nodeId2].color.z = 1.0;
						Node[nodeId2].drawNodeIs = true;
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
						if(AblateModeIs == true)
						{
							Node[i].isAblated = true;
							Node[i].drawNodeIs = true;
							Node[i].color.x = 1.0;
							Node[i].color.y = 1.0;
							Node[i].color.z = 1.0;
						}
						
						if(EctopicBeatModeIs == true)
						{
							PauseIs = true;
							printf("\n Node number = %d", i);
							setEctopicBeat(i);
						}
						
						if(AdjustMuscleAreaModeIs == true)
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
							
							Node[i].drawNodeIs = true;
							if(Node[i].isAblated == false) // If it is not ablated color it.
							{
								Node[i].color.x = 0.0;
								Node[i].color.y = 0.0;
								Node[i].color.z = 1.0;
							}
						}
						
						if(EctopicEventModeIs == true)
						{
							cudaMemcpy( Node, NodeGPU, NumberOfNodes*sizeof(nodeAtributesStructure), cudaMemcpyDeviceToHost);
							cudaErrorCheck(__FILE__, __LINE__);
							
							Node[i].isFiring = true; // Setting the ith node to fire the next time in the next time step.
							
							cudaMemcpy( NodeGPU, Node, NumberOfNodes*sizeof(nodeAtributesStructure), cudaMemcpyHostToDevice );
							cudaErrorCheck(__FILE__, __LINE__);
						}
						
						if(FindNodeModeIs == true)
						{
							Node[i].drawNodeIs = true;
							Node[i].color.x = 1.0;
							Node[i].color.y = 0.0;
							Node[i].color.z = 1.0;
							printf("\n Node number = %d", i);
						}
					}
				}
			}
		}
		else if(button == GLUT_RIGHT_BUTTON) // Right Mouse button down
		{
			if(AdjustMuscleLineModeIs == true)
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
					if(Node[nodeId1].isAblated == false)
					{
						Node[nodeId1].color.x = 0.0;
						Node[nodeId1].color.y = 1.0;
						Node[nodeId1].color.z = 0.0;
						Node[nodeId1].drawNodeIs = false;
					}
					
					if(Node[nodeId2].isAblated == false)
					{
						Node[nodeId2].color.x = 0.0;
						Node[nodeId2].color.y = 1.0;
						Node[nodeId2].color.z = 0.0;
						Node[nodeId2].drawNodeIs = false;
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
						Muscle[muscleId].isDisabled = false;
						
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
						if(AblateModeIs == true)
						{
							Node[i].isAblated = false;
							Node[i].drawNodeIs = false;
							Node[i].color.x = 0.0;
							Node[i].color.y = 1.0;
							Node[i].color.z = 0.0;
						}
						
						if(AdjustMuscleAreaModeIs == true)
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
									Muscle[muscleId].isDisabled = false;
									
									// Checking to see if the muscle needs to be killed.
									checkMuscle(muscleId);
								}
							}
							
							Node[i].drawNodeIs = true;
							if(Node[i].isAblated == false) // If it is not ablated color it.
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
		else if(button == GLUT_MIDDLE_BUTTON)
		{
			if(ScrollSpeedToggle == 0)
			{
				ScrollSpeedToggle = 1;
				ScrollSpeed = 1.0;
				printf("\n speed = %f\n", ScrollSpeed);
			}
			else
			{
				ScrollSpeedToggle = 0;
				ScrollSpeed = 0.1;
				printf("\n speed = %f\n", ScrollSpeed);
			}
			
		}
		drawPicture();
		copyNodesMusclesToGPU();
		//printf("\nSNx = %f SNy = %f SNz = %f\n", NodePosition[0].x, NodePosition[0].y, NodePosition[0].z);
	}
	
	if(state == 0)
	{
		if(button == 3) //Scroll up
		{
			MouseZ -= ScrollSpeed;
		}
		else if(button == 4) //Scroll down
		{
			MouseZ += ScrollSpeed;
		}
		//printf("MouseZ = %f\n", MouseZ);
		drawPicture();
	}
}
