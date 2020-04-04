function folder_creation_success = createFolder(folder_name)
    if (exist(folder_name,'dir') ~= 7)
        if(mkdir(folder_name) == 1)
            folder_creation_success = 1;
        end
    end
end

