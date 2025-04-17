/**
 * @file config.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-08-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef OPTIX_TRACER_CONFIG_H_INCLUDED
#define OPTIX_TRACER_CONFIG_H_INCLUDED

// Some global parameters
#define CHUNK_SIZE 16  // Chunk size for one traversal
#define STEP_EPSILON 0.00001

#define RGB_OFFSET 0 // 3
#define DEPTH_OFFSET 3 // 1
#define ACCUM_OFFSET 4 // 1
#define NORMAL_OFFSET 5  // 3
#define FINALT_OFFSET 8 // 1
#define NUM_CHANNELS_F 9 // Default 3, RGB


#define NUM_CHANNELS_I 1

#define BLOCK_X 16
#define BLOCK_Y 16

#endif
