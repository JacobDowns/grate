function export_issm_mesh(base_model_file, out_file)
% export_issm_mesh(base_model_file, out_file)
%   Load an ISSM Base_model.mat file, extract mesh information, and save it
%   in a simple, scipy-friendly .mat file.
%
% Example:
%   export_issm_mesh('Base_model.mat', 'mesh_info.mat')

    if nargin < 1
        error('Usage: export_issm_mesh(''Base_model.mat'',''mesh_info.mat'')');
    end
    if nargin < 2
        out_file = 'mesh_info.mat';
    end

    fprintf('Loading base model from: %s\n', base_model_file);
    S = load(base_model_file);

    if ~isfield(S, 'md')
        error('File %s does not contain variable ''md''.', base_model_file);
    end

    md   = S.md;
    mesh = md.mesh;

    % Basic required mesh fields
    elements = mesh.elements;          % (nelem x 3)
    x        = mesh.x;                 % (nvert x 1)
    y        = mesh.y;                 % (nvert x 1)
    base     = md.geometry.base;     % (nvert x 1)
    thickness = md.geometry.thickness; % (nvert x 1)
    surface = md.geometry.surface;   % (nvert x 1)
    bed = md.geometry.bed;           % (nvert x 1)
    lon = md.mesh.long;               % (nvert x 1)
    lat = md.mesh.lat;               % (nvert x 1)
    epsg = md.mesh.epsg;

    % Print a quick summary
    fprintf('\n--- Mesh summary ---\n');
    fprintf('  numberofelements : %d\n', mesh.numberofelements);
    fprintf('  numberofvertices : %d\n', mesh.numberofvertices);
    if ~isempty(epsg)
        fprintf('  epsg             : %d\n', epsg);
    end
    fprintf('  elements size    : %dx%d\n', size(elements,1), size(elements,2));
    fprintf('  x size           : %dx%d\n', size(x,1), size(x,2));
    fprintf('  y size           : %dx%d\n', size(y,1), size(y,2));

    % Save in a simple format that scipy.io.loadmat can read easily
    fprintf('\nSaving mesh to: %s\n', out_file);
    save(out_file, ...
         'elements', 'x', 'y', ...
         'lat', 'lon', 'epsg', ...
         'base', 'thickness', 'surface', 'bed', ...
         '-v7');

    fprintf('Done.\n');
end

