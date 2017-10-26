# DeepER-Lite
Entity Resolution in databases

## Running

First install [Docker](https://www.docker.com/), then:

    docker run -it -v /path/to/datasets/root/directory:/data daqcri/deeper-lite <Dataset> <first-table> <second-table> <perfect-mappings-file>

## Development

Edit source files then build a new image:

    docker build .
