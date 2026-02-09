function inspect_issm_mat(filename)
    % inspect_issm_mat(filename)
    % Load an ISSM-related .mat file and print useful information.
    %
    % Example (command line):
    %   matlab -nodisplay -nosplash -r "inspect_issm_mat('Base_model.mat'); exit"

    fprintf('Loading file: %s\n', filename);
    data = load(filename);

    fprintf('\nVariables in file:\n');
    disp(fieldnames(data));

    % If the file contains an ISSM model object (md), inspect structure
    if isfield(data, 'md')
        md = data.md;
        fprintf('\n--- Detected ISSM model object: md ---\n');

        % Print major components
        fprintf('Mesh fields:\n');
        if isfield(md.mesh, 'x')
            fprintf('  mesh.x: %d nodes\n', length(md.mesh.x));
        end
        if isfield(md.mesh, 'elements')
            fprintf('  mesh.elements: %d triangles\n', size(md.mesh.elements,1));
        end

        fprintf('\nGeometry fields:\n');
        gfields = fieldnames(md.geometry);
        fprintf('  %s\n', strjoin(gfields.', ', '));

        fprintf('\nMaterials fields:\n');
        mfields = fieldnames(md.materials);
        fprintf('  %s\n', strjoin(mfields.', ', '));

        fprintf('\nResults available?\n');
        if isfield(md, 'results') && isfield(md.results, 'TransientSolution')
            fprintf('  TransientSolution entries: %d\n', length(md.results.TransientSolution));
        else
            fprintf('  No results found in md.results\n');
        end

    else
        fprintf('\n--- No ''md'' model object found. Inspecting raw variables ---\n\n');
        vars = fieldnames(data);
        for k = 1:length(vars)
            vname = vars{k};
            v = data.(vname);
            fprintf('Variable %s: ', vname);
            disp(size(v));
        end
    end

    fprintf('\nDone.\n');
end

