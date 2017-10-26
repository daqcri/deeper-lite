# DeepER-Lite
Entity Resolution in databases

## Running

First install [Docker](https://www.docker.com/), then:

    docker run -it -v /path/to/datasets/root/directory:/data daqcri/deeper-lite <Dataset> <first-table> <second-table> <perfect-mappings-file>

## Development

Get the source:

    git clone --recursive git@github.com:daqcri/deeper-lite.git
    
Edit source files then build a new image:

    docker build .

Then run:

    docker run -it -v /data/dir:/data <image-id> <Dataset> ... # same like above

