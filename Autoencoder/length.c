#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (argc < 2){
        puts("Don't have filename");
        return 0;
    }
    size_t line_length = 80;
    char *s;
    
    if(argc > 2) {
        line_length = strtoul(argv[2], &s, 10);
    }
    
    FILE *pf = fopen(argv[1], "r");
    if(pf == NULL) {
        puts("ERROR: file not found");
        return 0;
    }
    size_t length = 0, line_number = 1;
    char str[line_length + 1];
    str[line_length] = '\0';
    do {
        str[length] = fgetc(pf);
        if (str[length] == '\t' || str[length] == ' ') {
            continue;
        }
        if (length >= line_length-1) {
            printf("Line %d has length %d\n\t%s ...\n", 
                   line_number, 
                   length+1, 
                   str
            );
            length = 0;
            char c;
            while((c = fgetc(pf)) != '\n' && c != EOF);
            line_number++;
            continue;
        }
        if (str[length] == '\n') {
            line_number++;
            length = -1;
        }
        length++;

    } while(!feof(pf));

    return 0;
}