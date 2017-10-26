# DeepER-Lite
Entity Resolution in databases

## Installation

First install [Docker](https://www.docker.com/), then:

    docker build .
    
## Running

    docker run -it <image> -v /path/to/datasets/root/directory:/data <Dataset> <first-table> <second-table> <perfect-mappings-file>
