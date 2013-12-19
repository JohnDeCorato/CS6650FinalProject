#define N_FOR_VIS (900)
#define DT .2
#define VISUALIZE 1
#define SOFTENING_FACTOR .0001
#define SHELL_NUM 2
#define INSTACING 0

// ========
#define RANDOM_POS 1
#define WALLS 2
#define POS_MODE WALLS

// ========
// MATRIX for matrix, NAIVE for n^2 version
#define MATRIX 1
#define NAIVE 2
#define COMPARE_MODE MATRIX

// ========
#define BOIDS 2
#define STEER 3
#define RUN_MODE STEER

// ========
#define COLOR_MIXED 1
#define COLOR_DIVIDED 2
#define COLOR_MODE COLOR_MIXED

#define PLANET_MASS 3e8
#define STAR_MASS 5e10
#define SCENE_SCALE 1e2 //size of the height map in simulation space


// BOIDS CONSTANTS
#define ATTRACTION_RADIUS 30.0


// OPENSTEER Library Constants
#define AVOID_RADIUS 7.0
#define OBJECT_RADIUS 1.0
#define MAX_SPEED 1
#define MAX_FORCE 1000
#define AGENT_MASS 1.0f

// AGENT STARTING POSITION PARAMETERS
#define LINES_MIDDLE_SEP 100.0
#define LINES_ROW_SEP 15.0
#define LINES_COL_SEP 15.0
#define NUM_PER_COLUMN 30

#define NUM_PER_WIDTH 

#define BLOCK_SIDE_SIZE 16
#define BLOCK_SIZE (BLOCK_SIDE_SIZE*BLOCK_SIDE_SIZE)
