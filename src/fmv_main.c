#include "fvm_linalg.h"
#include "fvm.h"
#include "fvm_mat.h"
#include "fvm_vec.h"
#include <stdio.h>
#include <string.h>

int in(char **arr, int len, char *target);

int main()
{
    char config[256];
    const char delimiters[] = ";,";
    char* config_token;
    char* saveptr_config = NULL;

    const int n_modes = 1;
    const char* modelist[] = {"test"};
    const char* mode = NULL;

    if (fgets(config, sizeof(config), stdin) != NULL)
    {
        config_token = strtok_r(config, delimiters, &saveptr_config);
        while (config_token != NULL)
        {   
            if (in(modelist, n_modes, config_token) == 1)
            {
                mode = config_token;
            }
            config_token = strtok_r(NULL, delimiters, &saveptr_config);
        }
        
        if (mode != NULL)
        {
            printf("Program runs in %s mode\n", mode);
        }
        else
        {
            printf("No valid mode found\n");
        }
    }
    return 0;
}

int in(char **arr, int len, char *target) {
  int i;
  for(i = 0; i < len; i++) {
    if(strncmp(arr[i], target, strlen(target)) == 0) {
      return 1;
    }
  }
  return 0;
}