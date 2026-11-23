-- What a link allows follows from its resource (`lib.share_links.ShareTarget.access`), so it is not stored.
ALTER TABLE share_links DROP COLUMN access;
