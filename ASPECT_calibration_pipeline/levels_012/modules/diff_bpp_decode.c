/* 
 * Tomas Kasparek, 2022, BUT
 */

/*
 * call diff_bpp_decode WIDTH HEIGHT GAPS OFF0,OFF1,OFF2 <input.diff.raw >output.raw
 *
 * Decode the differentialy encoded data cube (inverse to diff_bpp)
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>

#define MAX_WIDTH       1024
#define MAX_HEIGHT      1024
#define MAX_GAPS        32
#define BYTES_PER_PIXEL 2
#define MAX_ROW_BYTES   (MAX_WIDTH * BYTES_PER_PIXEL)
#define RES_OK          0

//program parameters
uint16_t  width = 0;
uint16_t  height = 0;
uint16_t  gaps = 0;

//global static buffer
uint16_t input_img[MAX_GAPS*MAX_HEIGHT*MAX_WIDTH];  // buffer to read the whole raw image, for 16bit 2048x2048 this is 8MB
int32_t  gaps_off[MAX_GAPS];            //list of gaps offsets, the first have to be 0 (no offset of the first image to itself)

int main(int argc, char **argv) {
  // size_t len=0;
  uint16_t x=0, y=0, g=0;
  char *off_str=NULL;
  uint16_t decoded_val=0;
 
  if (argc < 8) {
    fprintf(stderr, "Usage: %s -w <width> -h <height> -g <gaps> <offsets>\n", argv[0]);
    return 1;
  }

  width = (uint16_t)atoi(argv[2]);    // argv[1] == "-w"
  height = (uint16_t)atoi(argv[4]);   // argv[3] == "-h"
  gaps = (uint16_t)atoi(argv[6]);     // argv[5] == "-g"

  if(gaps <2){
    fprintf(stderr, "Need at least 2 gaps!\n");
    return 1;
  }

  off_str = argv[7];
  for (g = 0; g < gaps; g++) {
      gaps_off[g] = atoi(off_str);
      off_str = strchr(off_str, ',');
      if (off_str != NULL) off_str++;
  }

  fprintf(stderr, "w: %d, h: %d, g: %d, o: %s\n", width, height, gaps, argv[7]);

  //read whole image from input data
  bzero(&input_img, MAX_GAPS*MAX_HEIGHT*MAX_ROW_BYTES);
  // if((len = read(0, &input_img, gaps*width*height*BYTES_PER_PIXEL)) != gaps*width*height*BYTES_PER_PIXEL){
  //   fprintf(stderr, "Unable to read image got %ld B\n", len);
  //   return 1;
  // }
  size_t expected_bytes = gaps * width * height * BYTES_PER_PIXEL;
  size_t total_read = 0;
  uint8_t *buf = (uint8_t *)input_img;
  while (total_read < expected_bytes) {
      ssize_t n = read(0, buf + total_read, expected_bytes - total_read);
      if (n <= 0) {
          fprintf(stderr, "Read failed or incomplete: got %zu of %zu bytes\n", total_read, expected_bytes);
          return 1;
      }
      total_read += n;
  }
  //output first gap as is - no difference
  write(1, input_img, width*height*BYTES_PER_PIXEL);

  for(g=1; g < gaps; g++){

    //generate diff image for this gap
    for(y=0; y < height; y++){
      for (x=0; x < width; x++){
        decoded_val =  input_img[(g-1)*(width*height)+y*width+x] - input_img[g*(width*height)+y*width+x] - gaps_off[g];

        //store this decoded pixel for next wvl to be used
        input_img[g*(width*height)+y*width+x] = decoded_val;

        //output this decoded pixel
        write(1, &decoded_val, sizeof(uint16_t));
      }
    }
  }

  return 0;
}
